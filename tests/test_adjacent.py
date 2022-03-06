from pathlib import Path

import pytest

from mdeq.adjacent import (
    sequential_groups,
)


DATADIR = Path(__file__).parent / "data"


def test_sequential_groups():
    # no data
    with pytest.raises(ValueError):
        sequential_groups([], 1)

    data = list(range(4))
    # invalid num
    with pytest.raises(ValueError):
        sequential_groups(data, 0)

    with pytest.raises(ValueError):
        sequential_groups(data, -1)

    with pytest.raises(ValueError):
        sequential_groups(data, 6)

    # valid input
    got = sequential_groups(data, 1)
    assert got == ((0,), (1,), (2,), (3,))
    got = sequential_groups(data, 2)
    assert got == ((0, 1), (1, 2), (2, 3))
    got = sequential_groups(data, 3)
    assert got == ((0, 1, 2), (1, 2, 3))
    got = sequential_groups(data, 4)
    assert got == ((0, 1, 2, 3),)


