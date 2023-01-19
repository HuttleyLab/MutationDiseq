import pathlib

import pytest

from cogent3 import get_app, make_aligned_seqs
from numpy.testing import assert_allclose

from mdeq.utils import (
    CompressedValue,
    configure_parallel,
    convert_db_to_new_sqlitedb,
    estimate_freq_null,
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
    serialiser = get_app("pickle_it") + get_app("compress")
    cv = CompressedValue(None)
    assert isinstance(cv.decompressed, bytes) and not cv.decompressed
    data = "abcdefghijhkly" * 10
    cv = CompressedValue(serialiser(data))
    assert isinstance(cv.decompressed, bytes)
    assert cv.as_primitive == data


def test_compressed_value_deserialises():
    from cogent3 import get_app

    to_prim = get_app("to_primitive")
    pickler = get_app("pickle_it")
    compressor = get_app("compress")

    serialiser = to_prim + pickler + compressor

    # deserialises if data just json
    data = dict(a=24, b=[0, 1], c="ACGTT")
    ser = serialiser(data)
    cv = CompressedValue(ser)
    assert isinstance(cv.as_primitive, dict)
    assert cv.as_primitive == data


@pytest.fixture(scope="session")
def hedenfalk():
    return [
        float(v) for v in (DATADIR / "hedenfalk_pvals.txt").read_text().splitlines()
    ]


@pytest.fixture(scope="session")
def ten_pvals():
    return (0.0, 0.0, 0.0, 0.043, 0.202, 0.333, 0.385, 0.515, 0.598, 0.617)


def test_est_freq_null(hedenfalk):
    """compare results to expected from R package"""
    # data file derived from R qvalue package
    got = estimate_freq_null(hedenfalk, use_mse=False)
    assert round(got, 3) == 0.669
    got = estimate_freq_null(hedenfalk, use_log=True, use_mse=False)
    assert round(got, 3) == 0.669
    got = estimate_freq_null(hedenfalk, use_mse=True)
    assert 0.0 < got < 1.0


def test_est_freq_null_2(ten_pvals):
    """compare results to expected from R package"""
    # data file derived from R qvalue package
    got = estimate_freq_null(ten_pvals, use_mse=False)
    assert round(got, 2) == 0.49
    got = estimate_freq_null(ten_pvals, use_log=True, use_mse=False)
    assert round(got, 2) == 0.48


def test_est_freq_null_boostrap(ten_pvals):
    """compare results to expected from R package"""
    # data file derived from R qvalue package
    got = estimate_freq_null(ten_pvals, use_mse=True)
    assert_allclose(got, 0.6315789)


def test_est_freq_null_gt1(hedenfalk):
    # shift distribution so estimate would be > 1
    data = hedenfalk[:]
    data = list(sorted(data))
    for i in range(len(data) - 20):
        data[i] = 1.0
    e = estimate_freq_null(data)
    assert e == 1.0


@pytest.mark.parametrize(
    "start,stop,step",
    (
        (-0.1, 0.8, 0.05),
        (0.1, -0.8, 0.05),
        (0.1, 0.8, -0.05),
        (0.1, 1.1, 0.05),
        (0.8, 0.1, 0.05),
    ),
)
def test_est_freq_null_invalid_range(ten_pvals, start, stop, step):
    with pytest.raises(ValueError):
        estimate_freq_null(ten_pvals, start=start, stop=stop, step=step)


@pytest.fixture(scope="function")
def tmp_dir(tmpdir_factory):
    return pathlib.Path(tmpdir_factory.mktemp("convert"))


def test_db_conversion(tmp_dir):
    from cogent3.app.io_new import (
        decompress,
        from_primitive,
        load_db,
        unpickle_it,
    )

    source = DATADIR / "fg_GSN_synthetic-lo_lo-300bp-1rep.sqlitedb"
    source = DATADIR / "4otu-aligns.sqlitedb"
    source = DATADIR / "toe-300bp.sqlitedb"

    dest = tmp_dir / source.name
    got = convert_db_to_new_sqlitedb(source, dest)
    deserialiser = decompress() + unpickle_it() + from_primitive()
    loader = load_db(deserialiser=deserialiser)
    obj = loader(got.completed[0])
    got.summary_not_completed
