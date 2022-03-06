from pathlib import Path

import pytest

from mdeq.adjacent import (
    make_identifier,
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


def test_make_identifier():
    a = make_unaligned_seqs({"a": "ACGG"}, info=dict(source="blah/a.json"))
    b = make_unaligned_seqs({"a": "ACGG"}, info=dict(source="blah/b.json"))
    c = make_unaligned_seqs({"a": "ACGG"}, info=dict(source="blah/c.json"))
    d = make_unaligned_seqs({"a": "ACGG"})

    # missing identifier
    with pytest.raises(ValueError):
        make_identifier([a, b, d])

    with pytest.raises(ValueError):
        make_identifier([d, a])

    got = make_identifier([a])
    assert got == "a"
    got = make_identifier([a, c])
    assert got == "a--c"
    got = make_identifier([a, b, c])
    assert got == "a--b--c"


