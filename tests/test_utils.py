import pytest
from cogent3 import make_aligned_seqs

from kath_library.utils.utils import get_foreground


@pytest.fixture()
def aln():

    _seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    seqs = make_aligned_seqs(_seqs, moltype="dna")
    return seqs


def test_doesnt_fail(aln):

    fg_edge = get_foreground(aln)
    print(fg_edge)
