import pathlib

import pytest

from cogent3 import make_aligned_seqs

from mdeq.utils import (
    CompressedValue,
    configure_parallel,
    foreground_from_jsd,
    get_foreground,
    paths_to_sqlitedbs_matching,
    set_fg_edge,
)


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def aln():
    _seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
        "Gorilla": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }
    return make_aligned_seqs(_seqs, moltype="dna")


def test_get_foreground(aln):
    """returns None if no key in info."""
    fg_edge = get_foreground(aln)
    assert fg_edge is None


def test_foreground_from_jsd(aln):
    """with no foreground, identifies ingrouped lineage."""
    # but fails if > 3 sequences
    with pytest.raises(NotImplementedError):
        foreground_from_jsd(aln)

    aln = aln.take_seqs(["Human", "Bandicoot", "Rhesus"])
    fg_edge = foreground_from_jsd(aln)
    assert fg_edge is not None


def test_configure_par():
    got = configure_parallel(False, 0)
    assert got == dict(parallel=False, par_kw=None)
    got = configure_parallel(False, 3)
    assert got == dict(parallel=True, par_kw=dict(max_workers=3, use_mpi=True))
    got = configure_parallel(True, 3)
    assert got == dict(parallel=True, par_kw=dict(max_workers=3, use_mpi=True))
    got = configure_parallel(True, 0)
    assert got == dict(parallel=True, par_kw=None)


def test_set_fg_edge():
    from cogent3.app.composable import NotCompleted

    data = dict(a="ACCGG", b="ACCGG", c="ACCGG")
    aln = make_aligned_seqs(data, info=dict(source="blah"))
    app = set_fg_edge(fg_edge="c")
    got = app(aln)
    assert got.info.fg_edge == "c"

    # returns NotCompleted with message ValueError if no fg_edge value provided
    app = set_fg_edge(fg_edge=None)
    got = app(aln)
    assert "ValueError" in got.message

    # fg_edge not in alignment
    app = set_fg_edge(fg_edge="d")
    got = app(aln)
    assert isinstance(got, NotCompleted)

    # calling with NotCompleted just returns the same NotCompleted
    got2 = app(got)
    assert got2 is got


def test_paths_to_sqlitedbs_matching():
    # test root has no sqlitedb
    # check this with recurse False
    got = paths_to_sqlitedbs_matching(DATADIR.parent, "", False)
    assert got == []
    # allowing recurse should match result of just using Pathlib
    got = {p.name for p in paths_to_sqlitedbs_matching(DATADIR.parent, "", True)}
    expect = {p.name for p in DATADIR.glob("*sqlitedb")}
    assert got == expect
    # specifying pattern should match hand-coded result
    got = {p.name for p in paths_to_sqlitedbs_matching(DATADIR.parent, "4otu*", True)}
    expect = {p.name for p in DATADIR.glob("4otu*.sqlitedb")}
    assert got == expect


def test_compressed_value_decompressed():
    from blosc2 import compress

    # returns empty string if no data
    cv = CompressedValue(None)
    assert isinstance(cv.decompressed, bytes) and not cv.decompressed

    # returns bytes if data not compressed
    data = "abcdefghijhkly" * 10
    cv = CompressedValue(data)
    assert cv.data == data
    assert isinstance(cv.decompressed, bytes)
    assert cv.decompressed == data.encode("utf8")
    # and if data was bytes
    cv = CompressedValue(data.encode("utf8"))
    assert cv.data == data.encode("utf8")
    assert isinstance(cv.decompressed, bytes)
    assert cv.decompressed == data.encode("utf8")

    # and if data is compressed
    cv = CompressedValue(compress(data.encode("utf8")))
    assert cv.data != data
    assert isinstance(cv.decompressed, bytes)
    assert cv.decompressed.decode("utf8") == data


def test_compressed_value_deserialises():
    import json
    import pickle

    from blosc2 import compress

    # deserialises if data just json
    data = dict(a=24, b=[0, 1], c="ACGTT")
    j_ser = json.dumps(data).encode("utf8")
    cv = CompressedValue(j_ser)
    assert isinstance(cv.deserialised, dict)
    assert cv.deserialised == data
    # and if compressed json
    cv = CompressedValue(compress(j_ser))
    assert isinstance(cv.deserialised, dict)
    assert cv.deserialised == data

    # deserialises if data just pickled
    p_ser = pickle.dumps(data)
    cv = CompressedValue(compress(p_ser))
    assert isinstance(cv.deserialised, dict)
    assert cv.deserialised == data
    # and if compressed pickled
    cv = CompressedValue(compress(p_ser))
    assert isinstance(cv.deserialised, dict)
    assert cv.deserialised == data
