import pytest

from cogent3 import make_aligned_seqs

from mdeq.utils import (
    configure_parallel,
    foreground_from_jsd,
    get_foreground,
    set_fg_edge,
)


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


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
