import pytest
from cogent3 import make_aligned_seqs
from cogent3.app import evo
from kath_library.jsd import get_entropy, get_jsd


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
        "Human":        "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bandicoot":    "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Rhesus":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
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


def test_get_jsd_max(aln):
    """
    checks that get_jsd() with edge="max" returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated.
    checks that the jsd is >= then that of the ingroup.
    """
    edge, ingroup, jsd = get_jsd(aln, evaluate="max")
    assert 0 <= jsd < 1
    assert edge in ingroup

    _edge, _ingroup, _jsd = get_jsd(aln)
    assert jsd >= _jsd


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
    _, _, jsd = get_jsd(single_nt_aln, evaluate="max")
    assert jsd == 0.0
    _, _, jsd = get_jsd(single_nt_aln, evaluate="total")
    assert jsd == 0.0

@pytest.fixture()
def diff_nt_aln():

    _seqs = {
        "Human":        "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "Bandicoot":    "TTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "Rhesus":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    return seqs

def test_jsd_diff_nt(diff_nt_aln):
    edge, _, jsd = get_jsd(diff_nt_aln, evaluate="ingroup")
    assert jsd == 0.0
    _, _, jsd = get_jsd(diff_nt_aln, evaluate="max")
    assert jsd == 1.0
    _, _, jsd = get_jsd(diff_nt_aln, evaluate="total")
    assert jsd == 1.0


def test_get_entropy_stat_pi(aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(aln)
    entropy = get_entropy(fitted, stat_pi=True, edge="Human")
    assert entropy > 0


def test_get_entropy(aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(aln)
    entropy = get_entropy(fitted, stat_pi=False, edge="Human")
    assert entropy > 0

