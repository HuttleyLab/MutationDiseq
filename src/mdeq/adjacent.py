"""ordered grouping of alignments for EOP testing"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from cogent3 import _Table as Table
from cogent3.app.composable import SERIALISABLE_TYPE, appify
from cogent3.app.io import get_data_store, load_db


T = TypeVar("T")


def physically_adjacent(
    table: Table, sample_ids: set[str]
) -> tuple[tuple[str, ...], ...]:
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

    adjacent = [pair for pair in all_adjacent if set(pair).issubset(sample_ids)]
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


@dataclass
class grouped_data:
    elements: tuple[T, ...]
    source: str

    def __post_init__(self):
        self.elements = tuple(self.elements)
        # make sure all alignments have exactly the same sequence names
        names = set(self.elements[0].names)
        for e in self.elements:
            assert set(e.names) == names, f"names {e.names} != {names}"


_loader = load_db()


@appify(SERIALISABLE_TYPE, SERIALISABLE_TYPE)
def load_data_group(
    data_store_path, data_identifiers: tuple[str] = None
) -> grouped_data:
    """

    Parameters
    ----------
    data_store_path : str
        path to a tinydb
    data_identifiers
        series of identifiers

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
        data_objs.append(obj)

    identifier = make_identifier(data_objs)
    for n, obj in zip(identifier.split("--"), data_objs):
        obj.info.name = n

    return grouped_data(elements=tuple(data_objs), source=identifier)
