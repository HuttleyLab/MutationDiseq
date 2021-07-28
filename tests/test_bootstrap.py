from cogent3 import make_aligned_seqs
from kath_library.bootstrap import confidence_interval
from kath_library.convergence import _get_convergence
from kath_library.t50 import _get_t50
import pytest

@pytest.fixture()
def aln():
    _seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    seqs.info["fg_edge"] = "Human"

    return seqs


def test_confidence_interval_with_convergence(aln):
    conf_int = confidence_interval(_get_convergence, 1)
    results = conf_int.run(aln)

    assert isinstance(results["observed"]["convergence"], float)
    assert isinstance(results[2]["convergence"], float)


def test_confidence_interval_with_t50(aln):
    conf_int = confidence_interval(_get_t50, 1)
    results = conf_int.run(aln)

    assert isinstance(results["observed"]["t50"], float)
    assert isinstance(results[2]["t50"], float)
