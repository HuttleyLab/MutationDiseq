import numpy
import pytest
from cogent3 import get_model, load_aligned_seqs, make_tree
from cogent3.app import evo
from kath_analysis.stationary_pi import get_stat_pi_via_brute, get_stat_pi_via_eigen
from numpy import eye
from numpy.testing import assert_allclose


@pytest.fixture()
def brca1_alignment():
    aln = load_aligned_seqs("~/repos/cogent3/tests/data/brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)
    return aln


@pytest.fixture()
def likelihood_gtr(brca1_alignment):
    """
    optimised GTR likelihood object
    """
    model_gtr = get_model("GTR", optimise_motif_probs=True)
    tree = make_tree(tip_names=brca1_alignment.names)

    lf_gtr = model_gtr.make_likelihood_function(tree)
    lf_gtr.set_alignment(brca1_alignment)

    lf_gtr.optimise(show_progress=False, max_restart=5, tolerance=1e-10)

    return lf_gtr


@pytest.fixture()
def likelihood_gn(brca1_alignment, likelihood_gtr):
    """
    optimised GS likelihood object initialised from GTR
    """

    gn = evo.model("GN", sm_args=dict(optimise_motif_probs=True), time_het="max")
    fitted_gn = gn(brca1_alignment)
    gn_lf = fitted_gn.lf

    gn_lf.initialise_from_nested(likelihood_gtr)
    gn_lf.optimise()

    return gn_lf


@pytest.fixture()
def almost_identity():
    P = numpy.array(
        [
            [
                0.9999999999473204,
                4.514833540937878e-15,
                5.161700461918482e-11,
                1.057951221582512e-12,
            ],
            [
                1.4896797655945398e-14,
                0.9999999999999828,
                1.5272937302688045e-15,
                5.676029024274522e-16,
            ],
            [
                7.453332476484502e-15,
                1.3956167360632593e-15,
                0.9999999999999851,
                6.22143477999565e-15,
            ],
            [
                9.561249256827964e-17,
                3.619348852029019e-15,
                5.121374414061707e-15,
                0.9999999999999912,
            ],
        ]
    )
    pi = numpy.array(
        [
            0.12773403671754724,
            0.30096238355059024,
            0.19247593282973588,
            0.37882764690212656,
        ]
    )
    return P, pi


@pytest.fixture()
def identity():
    P = eye(4)
    pi = numpy.array(
        [
            0.12773403671754724,
            0.30096238355059024,
            0.19247593282973588,
            0.37882764690212656,
        ]
    )
    return P, pi


def test_return_same_stat_pi(likelihood_gtr):
    """
    given a Psub matrix and pi defined by GTR, should return the same pi
    """
    pi = likelihood_gtr.get_motif_probs().to_array()
    psubs = likelihood_gtr.get_all_psubs().values()

    for p in psubs:
        p = p.to_array()
        assert_allclose(get_stat_pi_via_brute(p, pi), pi, rtol=1e-10, atol=1e-10)


def test_get_stat_pi(likelihood_gtr):
    """
    given a Psub matrix defined by GTR, and a pi that is logically not the stationary distribution,
    should return the pi that was defined by GTR
    """
    pi = likelihood_gtr.get_motif_probs().to_array()
    psubs = likelihood_gtr.get_all_psubs().values()

    pi_not_stat = numpy.zeros(len(pi))
    pi_not_stat[0] = 1

    for p in psubs:
        p = p.to_array()
        assert_allclose(get_stat_pi_via_brute(p, pi_not_stat), pi, rtol=1e-8, atol=1e-8)


def test_eigen_numerical_same_pi(likelihood_gn):
    """
    Given the same P matrix both the numerical calculation and EigenValue
    decomposition return the same value for pi_inf
    """

    pi = likelihood_gn.get_motif_probs().to_array()
    psubs = likelihood_gn.get_all_psubs().values()

    for p in psubs:
        p = p.to_array()
        numpy.testing.assert_almost_equal(
            get_stat_pi_via_brute(p, pi), get_stat_pi_via_eigen(p), decimal=8
        )


def test_throw_error():
    P = numpy.ones((4, 4))
    pi_0 = numpy.array([0.25] * 4)
    with pytest.raises(TypeError):
        get_stat_pi_via_brute(P, pi_0)


def test_eigen_with_almost_identity(almost_identity):
    P, pi = almost_identity
    get_stat_pi_via_eigen(P, pi)


def test_brute_with_almost_identity(almost_identity):
    P, pi = almost_identity
    get_stat_pi_via_brute(P, pi)


def test_eigen_with_identity(identity):
    P, pi = identity
    get_stat_pi_via_eigen(P, pi)


def test_brute_with_identity(identity):
    P, pi = identity
    get_stat_pi_via_brute(P, pi)
