import numpy
import pytest
from cogent3.app import evo, io
from numpy import array

from kath_library.convergence import convergence, eigII
from kath_library.utils.utils import get_pi_0, get_pi_tip

loader = io.load_db()


@pytest.fixture()
def mcr_dstore():
    dstore = io.get_data_store(
        f"/Users/katherine/repos/results/aim_2/synthetic/758_443154_73021/3000bp/mcr.tinydb"
    )

    return dstore


def test_convergence_GN(mcr_dstore):
    mc = loader(mcr_dstore[0])["mcr"]
    gn = loader(mcr_dstore[0])["mcr"]["GN"]

    fg_edge = mc.source["fg_edge"]

    pi = get_pi_tip(gn, fg_edge)
    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()

    t = gn.lf.get_param_value("length", edge=fg_edge)
    conv = convergence(pi, Q, t)

    assert conv >= 0


def test_convergence_non_zero(mcr_dstore):
    mc = loader(mcr_dstore[0])["mcr"]
    gn = loader(mcr_dstore[0])["mcr"]["GN"]

    fg_edge = mc.source["fg_edge"]

    pi = get_pi_tip(gn, fg_edge)
    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()

    new_pi = numpy.array([(pi[0] + pi[1]), pi[2] / 2, pi[2] / 2, pi[3]])
    t = gn.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(new_pi, Q, t)

    print(conv)
    assert conv >= 0


def test_convergence_GTR(mcr_dstore):
    mc = loader(mcr_dstore[0])["mcr"]
    gtr = loader(mcr_dstore[0])["mcr"]["GTR"]

    fg_edge = mc.source["fg_edge"]

    Q = gtr.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(gtr)

    t = gtr.lf.get_param_value("length", edge=fg_edge)
    conv = convergence(pi, Q, t)

    numpy.testing.assert_almost_equal(conv, 0, decimal=10)


def test_eigII():

    dstore = io.get_data_store(
        "~/repos/data/microbial/synthetic/758_443154_73021/3000bp.tinydb"
    )
    loader = io.load_db()
    aln1 = loader(dstore[0])
    fg_edge = aln1.info.fg_edge

    if fg_edge is None:
        raise TypeError("Alignment needs a info.fg_edge attribute")

    bg_edges = list({fg_edge} ^ set(aln1.names))

    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )

    result = GN(aln1)

    Q = result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)

    eigII(Q)
