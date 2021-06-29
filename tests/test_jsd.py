import pytest
from cogent3 import make_aligned_seqs
from cogent3.app import evo
from cogent3.app.result import generic_result
from kath_library.jsd import _jsd_fg_edge, _max_jsd_tree, get_entropy, get_jsd


@pytest.fixture()
def simple_aln():

    _seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    return seqs


def test_get_jsd_none_edge(simple_aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    edge, ingroup, jsd = get_jsd(simple_aln, edge=None)
    assert 0 <= jsd < 1
    assert edge in ingroup


def test_get_jsd_max_edge(simple_aln):
    """
    checks that get_jsd() with edge="max" returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated.
    checks that the jsd is >= then that of the ingroup.
    """
    edge, ingroup, jsd = get_jsd(simple_aln, edge="max")
    assert 0 <= jsd < 1
    assert edge in ingroup

    _edge, _ingroup, _jsd = get_jsd(simple_aln)
    assert jsd >= _jsd


def test_get_jsd_given_edge(simple_aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    edge, ingroup, jsd = get_jsd(simple_aln, edge="Human")
    assert 0 <= jsd < 1
    assert edge in ingroup


def test_get_jsd_given_edge_exception(simple_aln):
    """
    checks that get_jsd() with edge=None returns a value for
    JSD between 0 and 1, and that the foreground edge is in
    the tuple from which the JSD is calculated, the ingroup.
    """
    with pytest.raises(AssertionError):
        _, _, _ = get_jsd(simple_aln, edge="Katherine")


def test_get_entropy_stat_pi(simple_aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(simple_aln)
    entropy = get_entropy(fitted, stat_pi=True, edge="Human")
    assert entropy > 0


def test_get_entropy(simple_aln):
    gn = evo.model("GN", show_progress=False)
    fitted = gn(simple_aln)
    entropy = get_entropy(fitted, stat_pi=False, edge="Human")
    assert entropy > 0


def test_jsd_fg_edge(simple_aln):
    result = _jsd_fg_edge(simple_aln)
    assert isinstance(result, generic_result)
    assert result["jsd"] >= 0


def test_max_jsd_tree(simple_aln):
    result = _max_jsd_tree(simple_aln)
    assert isinstance(result, generic_result)
    assert result["jsd"] >= 0
