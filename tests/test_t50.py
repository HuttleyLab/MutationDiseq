import numpy
import pytest
from cogent3 import get_model, load_aligned_seqs, make_tree
from cogent3.maths.measure import jsm
from numpy.ma.core import dot
from scipy.linalg import expm

from kath_library.t50 import T50
from kath_library.utils.numeric_utils import (
    valid_probability_vector,
    valid_rate_matrix,
    valid_stochastic_matrix,
)


@pytest.fixture()
def t50_construction_random():
    """
    T50 object constructed with randomly populated Q and pi_0 = [0.25, 0.25, 0.25, 0.25]
    """
    import numpy

    Q = numpy.random.randint(1, 20, size=(4, 4))
    diag_indices = numpy.diag_indices(4)
    Q[diag_indices] = 0
    row_sum = Q.sum(axis=1)
    Q[diag_indices] = -row_sum
    print(Q)

    pi_0 = numpy.array([0.25] * 4)

    calc_t50 = T50(Q, pi_0, jsm)
    return calc_t50


@pytest.fixture()
def gtr_defined_t50():
    """
    T50 object constructed by stationary process (Q and pi defined GTR)
    """
    aln = load_aligned_seqs("~/repos/cogent3/tests/data/brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)

    model_gtr = get_model("GTR", optimise_motif_probs=True)
    tree = make_tree(tip_names=aln.names)

    lf_gtr = model_gtr.make_likelihood_function(tree)
    lf_gtr.set_alignment(aln)
    lf_gtr.optimise(
        show_progress=False, max_restart=5, tolerance=1e-10, limit_action="raise"
    )

    Q = lf_gtr.get_rate_matrix_for_edge("TombBat", calibrated=False).to_array()
    pi = lf_gtr.get_motif_probs().to_array()

    return T50(Q, pi)


@pytest.fixture()
def non_stationary_t50():
    """
    T50 object constructed by stationary process (Q and pi defined GTR)
    """
    aln = load_aligned_seqs("~/repos/cogent3/tests/data/brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)

    model_gtr = get_model("GTR", optimise_motif_probs=True)
    tree = make_tree(tip_names=aln.names)

    lf_gtr = model_gtr.make_likelihood_function(tree)
    lf_gtr.set_alignment(aln)
    lf_gtr.optimise(
        show_progress=False, max_restart=5, tolerance=1e-10, limit_action="raise"
    )

    Q = lf_gtr.get_rate_matrix_for_edge("TombBat", calibrated=False).to_array()
    pi = lf_gtr.get_motif_probs().to_array()
    new_pi = numpy.array([(pi[0] + pi[1]), pi[2] / 2, pi[2] / 2, pi[3]])

    return T50(Q, new_pi)


def test_construction():
    import numpy

    Q = numpy.random.randint(1, 20, size=(4, 4))
    diag_indices = numpy.diag_indices(4)
    Q[diag_indices] = 0
    row_sum = Q.sum(axis=1)
    Q[diag_indices] = -row_sum
    print(Q)

    pi_0 = numpy.array([0.25] * 4)

    calc_t50 = T50(Q, pi_0)
    print(calc_t50(0.2))

    t50 = calc_t50.estimate_t50()
    print(t50)


def test_stat_pi_correct_construction(t50_construction_random):
    assert valid_probability_vector(t50_construction_random.pi_inf)


def test_pi0_correct_construction(t50_construction_random):
    assert valid_probability_vector(t50_construction_random.pi_0)


def test_Q_correct_construction(t50_construction_random):
    assert valid_rate_matrix(t50_construction_random.Q)


def test_expm_Q_valid_stochastic_matrix(t50_construction_random):
    """
    the matrix exponential of Q is a valid stochastic matrix
    """
    assert valid_stochastic_matrix(expm(t50_construction_random.Q))


def test_dist_with_stationary_pi0(gtr_defined_t50):
    """
    distance to stationary pi given a stationary pi should be 0
    """
    dist = gtr_defined_t50.distance_from_pi_zero(gtr_defined_t50.pi_0)
    numpy.testing.assert_almost_equal(dist, 0, decimal=6)


def test_pi0_for_stationary_process(gtr_defined_t50):
    """
    for a stationary process, tau should scale Q such that the dot product between pi_0 and
    the resulting P matrix gives pi_0 (or extremely close).
    """

    Q = gtr_defined_t50.Q
    pi_0 = gtr_defined_t50.pi_0
    t50 = gtr_defined_t50.estimate_t50()

    tau = gtr_defined_t50.tau
    pi_inf_tau = dot(pi_0, expm(Q * tau))
    pi_inf = dot(pi_0, expm(Q))

    numpy.testing.assert_almost_equal(pi_inf_tau, pi_inf, decimal=6)


def test_T50_with_stationary_pi0(gtr_defined_t50):
    """
    T50 given a stationary pi should be 0
    """
    t50 = gtr_defined_t50.estimate_t50()
    print("t50 is", t50)
    numpy.testing.assert_almost_equal(
        t50, 0, decimal=7, err_msg=f"{gtr_defined_t50.tau}"
    )


def test_T50_with_non_stationary_pi(non_stationary_t50):
    t50 = non_stationary_t50.estimate_t50()
    print("t50 is", t50)
    assert t50 > 0


def test_t50_number_precision():
    pi = numpy.array(
        [
            0.4398948756903677,
            0.1623791467423164,
            0.31844113569205656,
            0.07928484187525932,
        ]
    )

    Q = numpy.array(
        [
            [
                -0.2157580748699195,
                0.14087587706637505,
                0.05629993490798896,
                0.0185822628955555,
            ],
            [
                0.38164122470865824,
                -0.5239099450725678,
                0.04540787875756386,
                0.09686084160634569,
            ],
            [
                0.07777278150293754,
                0.02315433460569617,
                -0.15227099863388555,
                0.051343882525251856,
            ],
            [
                0.1030996850488375,
                0.1983756344942528,
                0.2062185390733903,
                -0.5076938586164806,
            ],
        ]
    )
    get = T50(Q, pi)
    t50 = get.estimate_t50()
    print(t50)
