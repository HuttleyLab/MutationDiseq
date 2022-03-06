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

