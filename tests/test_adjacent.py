from pathlib import Path

import pytest

from cogent3 import make_unaligned_seqs, open_data_store
from cogent3.util.deserialise import deserialise_object

from mdeq.adjacent import (
    grouped,
    grouped_alignments,
    load_data_group,
    make_identifier,
    physically_adjacent,
    sequential_groups,
)
from mdeq.eop import ALT_AEOP, NULL_AEOP, adjacent_eop


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]


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
    got = make_identifier([a, a, a])
    assert got == "a--a--a"


def test_grouped_alignments():
    a = make_unaligned_seqs({"a": "ACGG", "b": "ACGG"}, info=dict(source="blah/a.json"))
    b = make_unaligned_seqs({"a": "ACGG", "b": "ACGG"}, info=dict(source="blah/b.json"))
    c = make_unaligned_seqs({"a": "ACGG", "d": "ACGG"}, info=dict(source="blah/c.json"))
    got = grouped_alignments([a, b])
    assert isinstance(got, grouped_alignments)
    with pytest.raises(AssertionError):
        grouped_alignments((a, c))

    # can group same alignment
    got = grouped_alignments((a, a))
    assert isinstance(got, grouped_alignments)


def test_grouped_data():
    got = grouped(identifiers=("a", "b"))
    assert isinstance(got, grouped)
    assert got.source == "a--b"
    assert got.identifiers == ("a", "b")
    assert (got[0], got[1]) == ("a", "b")

    # can group same alignment
    assert isinstance(got, grouped)
    got = grouped(identifiers=("a", "a"))
    assert got.source == "a--a"

    a = make_unaligned_seqs({"a": "ACGG", "b": "ACGG"}, info=dict(source="blah/a.json"))
    b = make_unaligned_seqs({"a": "ACGG", "b": "ACGG"}, info=dict(source="blah/b.json"))
    got.elements = [a, b]
    assert isinstance(got.elements, grouped_alignments)
    assert len(got.elements) == len(got.identifiers)

    json = got.to_json()
    # can we get it back
    obj = deserialise_object(json)
    assert obj.identifiers == got.identifiers
    for i in range(2):
        assert got.elements[i].to_dict() == obj.elements[i].to_dict()

    assert len(obj.elements) == len(got.elements)


def test_load_data_group():
    from random import choice, shuffle

    def eval_input(pair):
        got = group_loader(grouped(identifiers=pair))
        assert len(got.elements) == 2
        source = "--".join(e.replace(".json", "") for e in pair)
        assert got.source == source
        for i, n in enumerate(pair):
            o = got.elements[i]
            assert o.info.source == n.replace(".json", "")

    path = DATADIR / "300bp-new.sqlitedb"
    dstore = open_data_store(path)
    names = [m.unique_id for m in dstore]
    shuffle(names)

    paired = sequential_groups(names, 2)
    group_loader = load_data_group(str(path))
    # pair elements endwith json
    pair = choice(paired)
    # pair elements does not end with json
    eval_input(pair)
    eval_input(tuple(e.replace(".json", "") for e in pair))


def test_new_adjacent_eop():
    from random import choice, shuffle

    path = DATADIR / "300bp-new.sqlitedb"
    dstore = open_data_store(path)
    names = [m.unique_id for m in dstore]
    shuffle(names)

    paired = sequential_groups(names, 2)
    pair = grouped(e.replace(".json", "") for e in choice(paired))
    group_loader = load_data_group(str(path))
    data = group_loader(pair)
    eop = adjacent_eop(
        opt_args={"max_restarts": 1, "max_evaluations": 10, "limit_action": "ignore"}
    )
    got = eop(data)
    assert len(got[NULL_AEOP]) == 1
    assert len(got[ALT_AEOP]) == 2
    for k in got:
        assert isinstance(got[k].lnL, float)
        assert isinstance(got[k].nfp, int)

    # alt has more parameters than null
    assert got[ALT_AEOP].nfp > got[NULL_AEOP].nfp


def test_physically_adjacent():
    from cogent3 import load_table

    path = DATADIR / "gene_order.tsv"
    table = load_table(path)
    sample_ids = {
        "ENSG00000142657",
        "ENSG00000184677",
        "ENSG00000184454",  # follows above, but on separate chromosome
        "ENSG00000283761",
        "ENSG00000158477",
        "ENSG00000162739",
    }
    got = physically_adjacent(table, sample_ids)
    assert set(got) == {
        grouped(identifiers=p)
        for p in [
            ("ENSG00000142657", "ENSG00000184677"),
            ("ENSG00000158477", "ENSG00000162739"),
        ]
    }

    with pytest.raises(ValueError):
        physically_adjacent(table[:, ["name", "start"]], sample_ids)
