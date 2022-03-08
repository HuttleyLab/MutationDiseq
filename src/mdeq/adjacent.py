"""ordered grouping of alignments for EOP testing"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from cogent3 import _Table as Table
from cogent3.app.composable import SERIALISABLE_TYPE, appify
from cogent3.app.io import get_data_store, load_db
from cogent3.util import deserialise
from cogent3.util.misc import get_object_provenance

from mdeq.utils import SerialisableMixin


T = TypeVar("T")


class grouped_alignments(tuple):
    """a validating tuple"""

    def __new__(cls, elements):
        elements = tuple(elements)
        names = set(elements[0].names)
        for e in elements:
            assert set(e.names) == names, f"names {e.names} != {names}"
        return tuple.__new__(cls, elements)


@dataclass(eq=True, unsafe_hash=True)
class grouped(SerialisableMixin):
    identifiers: tuple[str, ...]
    source: str = None
    _elements: grouped_alignments = None

    def __post_init__(self):
        self.identifiers = tuple(self.identifiers)
        self.source = make_identifier(self.identifiers)

    def __getitem__(self, item):
        return self.identifiers[item]

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements):
        self._elements = grouped_alignments(elements)

    def to_rich_dict(self):
        result = super().to_rich_dict()
        result = {**result, **asdict(self)}
        result["_elements"] = [m.to_rich_dict() for m in self.elements]
        return result

    @classmethod
    def from_dict(cls, data):
        data.pop("type", None)
        elements = [deserialise.deserialise_object(e) for e in data.pop("_elements")]
        result = cls(**data)
        result.elements = elements
        return result


@deserialise.register_deserialiser(get_object_provenance(grouped))
def deserialise_grouped(data):
    return grouped.from_dict(data)


def physically_adjacent(table: Table, sample_ids: set[str]) -> tuple[grouped, ...]:
    """identifiers members of id_set that are adjacent in table

    Parameters
    ----------
    table
        cogent3 Table of all genes in a genome
    sample_ids
        sample ID's
    """

    all_adjacent = []
    for coord_name in table.distinct_values("coord_name"):
        sub_table = table.filtered(lambda x: x == coord_name, columns="coord_name")
        if sub_table.shape[0] == 1:
            continue
        all_adjacent.extend(sequential_groups(sub_table.columns["name"], 2))

    adjacent = [
        grouped(identifiers=pair)
        for pair in all_adjacent
        if set(pair).issubset(sample_ids)
    ]
    return tuple(adjacent)


def sequential_groups(data, num: int) -> tuple[tuple[T, ...], ...]:
    """returns all num sequential overlapping elements"""
    if num < 1:
        raise ValueError(f"{num=}, should be >= 1")
    if len(data) < num:
        raise ValueError(f"len of data {len(data)} < {num=}")
    return tuple(tuple(data[i : i + num]) for i in range(len(data) - num + 1))


def make_identifier(data) -> str:
    """identifies source for each element in data and makes ordered identifier,
    no suffix
    Returns
    -------
    e1--e2... (double hyphen between names without suffix)
    """
    from cogent3.app.data_store import get_data_source

    sources = [get_data_source(e) for e in data]
    if "unknown" in sources:
        raise ValueError(f"'unknown' source present in {sources}")

    composite = []
    for e in sources:
        e = Path(e)
        l = -len(e.suffix) or None
        composite.append(e.name[:l])

    return "--".join(composite)


_loader = load_db()


@appify(SERIALISABLE_TYPE, SERIALISABLE_TYPE)
def load_data_group(data_store_path, data_identifiers: grouped = None) -> grouped:
    """

    Parameters
    ----------
    data_store_path : str
        path to a tinydb
    data_identifiers
        grouped instance

    Notes
    -----
    Each data object has its identifier assigned to info.name attribute
    """
    dstore = get_data_store(data_store_path)
    data_objs = []
    for identifier in data_identifiers:
        identifier = (
            identifier if identifier.endswith(".json") else f"{identifier}.json"
        )
        m = dstore.filtered(identifier)
        assert len(m) == 1
        obj = _loader(m[0])
        if not obj:  # probably not completed error
            return obj
        obj.info.name = identifier.replace(".json", "")
        data_objs.append(obj)

    dstore.close()

    data_identifiers.elements = data_objs
    return data_identifiers
