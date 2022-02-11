import pytest

from cogent3 import make_aligned_seqs
from cogent3.app import evo

from mdeq.jsd import get_entropy, get_jsd
from numpy import allclose


@pytest.fixture()
def aln():

    _seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    return seqs


@pytest.fixture()
def single_nt_aln():

    _seqs = {
        "Human": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bandicoot": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Rhesus": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    }

    return make_aligned_seqs(_seqs, moltype="dna")


@pytest.fixture()
def diff_nt_aln():

    _seqs = {
        "Human": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bandicoot": "TTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "Rhesus": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    }

    return make_aligned_seqs(_seqs, moltype="dna")


@pytest.fixture()
def diff_nt_aln_with_fg():

    _seqs = {
        "Human": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bandicoot": "TTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "Rhesus": "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    seqs.info["fg_edge"] = "Human"

    return seqs


def test_get_jsd_none_edge(aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    edge, ingroup, jsd = get_jsd(aln, edge=None)
    assert 0 <= jsd < 1
    assert edge in ingroup


def test_get_jsd_all(aln):
    """
    checks that get_jsd() with evaluate="all" returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated.
    checks that the jsd is >= then that of the ingroup.
    """
    edge, ingroup, jsd = get_jsd(aln, evaluate="all")
    assert 0 <= jsd["ingroup_jsd"] < 1
    assert 0 <= jsd["total_jsd"] < 1
    assert edge in ingroup
    assert jsd["total_jsd"] >= jsd["ingroup_jsd"]


def test_get_jsd_max(aln):
    """
    checks that get_jsd() with evaluate="all" returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated.
    checks that the jsd is >= then that of the ingroup.
    """
    edge, ingroup, jsd = get_jsd(aln, evaluate="max")
    assert 0 <= jsd < 1
    assert edge in ingroup


def test_get_jsd_edge(aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    edge, ingroup, jsd = get_jsd(aln, edge="Human")
    assert 0 <= jsd < 1
    assert edge in ingroup


def test_get_jsd_given_edge_exception(aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    with pytest.raises(AssertionError):
        _, _, _ = get_jsd(aln, edge="Katherine")


def test_jsd_single_nt(single_nt_aln):
    _, _, jsd = get_jsd(single_nt_aln, evaluate="ingroup")
    assert jsd == 0.0
    _, _, jsd = get_jsd(single_nt_aln, evaluate="total")
    assert jsd == 0.0


def test_jsd_diff_nt(diff_nt_aln):
    edge, ingroup, jsd = get_jsd(diff_nt_aln, evaluate="ingroup")
    assert jsd == 0.0
    assert ingroup == ("Human", "Rhesus")
    _, _, jsd = get_jsd(diff_nt_aln, evaluate="total")
    # expected value computed manually
    assert allclose(jsd, 0.9182958340544896)


def test_jsd_diff_nt_with_fg(diff_nt_aln_with_fg):
    edge, _, jsd = get_jsd(diff_nt_aln_with_fg, evaluate="ingroup")
    assert jsd == 0.0
    _, _, jsd = get_jsd(diff_nt_aln_with_fg, evaluate="total")
    # expected value computed manually
    assert allclose(jsd, 0.9182958340544896)


def test_get_entropy(aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(aln)
    entropy = get_entropy(fitted, edge="Human", stat_pi=False)
    assert entropy > 0


def test_get_entropy_stat_pi(aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(aln)
    entropy = get_entropy(fitted, edge="Human", stat_pi=True)
    assert entropy > 0
