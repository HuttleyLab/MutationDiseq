import numpy

from cogent3 import make_aligned_seqs
from cogent3.app import evo
from cogent3.util.dict_array import DictArray
from numpy import eye

from mdeq.numeric import (
    valid_probability_vector,
    valid_rate_matrix,
    valid_stochastic_matrix,
)


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


def test_valid_rate_matrix_DictArray():
    Q = numpy.random.randint(1, 20, size=(4, 4))
    diag_indices = numpy.diag_indices(4)
    Q[diag_indices] = 0
    row_sum = Q.sum(axis=1)
    Q[diag_indices] = -row_sum
    d_array_Q = DictArray(Q)

    assert valid_rate_matrix(d_array_Q)


def test_valid_psub_DictArray():
    P = eye(4)
    d_array_P = DictArray(P)

    assert valid_stochastic_matrix(d_array_P)


def test_valid_probability_vector_DictArray():
    seqs = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    aln = make_aligned_seqs(seqs, moltype="dna")

    gn = evo.model("GN", show_progress=False)
    fitted = gn(aln)

    lf = fitted.lf
    pi = lf.get_motif_probs()

    assert valid_probability_vector(pi)
