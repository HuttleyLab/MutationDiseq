"""ordered grouping of alignments for EOP testing"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from cogent3.app.composable import SERIALISABLE_TYPE, appify
from cogent3.app.io import get_data_store, load_db


T = TypeVar("T")


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
