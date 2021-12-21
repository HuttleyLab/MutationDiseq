import os
import pathlib

from tempfile import TemporaryDirectory

import numpy
import pytest
from cogent3.app import io

from mdeq.bootstrap import create_bootstrap_app
from mdeq.convergence import (
    convergence,
    eigII,
    get_convergence,
    get_convergence_bstrap,
    get_convergence_mc,
)
from mdeq.model import GN_sm
from mdeq.utils.utils import get_foreground, get_pi_0, get_pi_tip

DATADIR = pathlib.Path(__file__).parent / "data"

loader = io.load_db()


@pytest.fixture()
def mcr_dstore():
    return io.get_data_store(DATADIR / "mcr.tinydb")


def test_convergence_GN(mcr_dstore):
    """
    Convergence for a GN process should be greater or equal to zero.
    """
    result = loader(mcr_dstore[0])
    fg_edge = result["fg_edge"]
    gn = result["mcr"]["GN"]

    pi = get_pi_tip(gn, fg_edge)
    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()

    t = gn.lf.get_param_value("length", edge=fg_edge)
    conv = convergence(pi, Q, t)

    assert conv >= 0


def test_convergence_non_zero(mcr_dstore):
    """
    Convergence of a non-stationary process should be greater than 0
    """
    result = loader(mcr_dstore[0])
    fg_edge = result["fg_edge"]
    gn = result["mcr"]["GN"]

    pi = get_pi_tip(gn, fg_edge)
    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()

    new_pi = numpy.array([(pi[0] + pi[1]), pi[2] / 2, pi[2] / 2, pi[3]])
    t = gn.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(new_pi, Q, t)

    assert conv >= 0


def test_convergence_GTR(mcr_dstore):
    """
    The Convergence of a stationary process should be 0
    """
    result = loader(mcr_dstore[0])
    fg_edge = result["fg_edge"]
    gtr = result["mcr"]["GTR"]

    Q = gtr.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(gtr)

    t = gtr.lf.get_param_value("length", edge=fg_edge)
    conv = convergence(pi, Q, t)

    numpy.testing.assert_almost_equal(conv, 0, decimal=10)


def test_eigII():

    dstore = io.get_data_store(DATADIR / "3000bp.tinydb")
    loader = io.load_db()
    aln1 = loader(dstore[0])
    fg_edge = get_foreground(aln1)

    if fg_edge is None:
        raise TypeError("Alignment needs a info.fg_edge attribute")

    bg_edges = list({fg_edge} ^ set(aln1.names))
    GN = GN_sm(discrete_edges=bg_edges)

    result = GN(aln1)

    Q = result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)

    eigII(Q)


def test_get_convergence(mcr_dstore):
    loader = io.load_db()
    gn = loader(mcr_dstore[0])["mcr"]["GN"]

    conv = get_convergence(gn)
    assert conv["convergence"] >= 0


def test_get_convergence_mc(mcr_dstore):
    loader = io.load_db()
    gn_mc = loader(mcr_dstore[2])
    conv = get_convergence_mc(gn_mc)
    assert conv["convergence"] >= 0


def test_get_convergence_mc_composable(mcr_dstore):
    with TemporaryDirectory(dir=".") as dirname:

        reader = io.load_db()

        outpath = os.path.join(os.getcwd(), dirname, "tempdir.tinydb")
        writer = io.write_db(outpath)

        process = reader + get_convergence_mc + writer

        process.apply_to(mcr_dstore[:1])
        assert len(process.data_store.summary_incomplete) == 0


def test_get_convergence_bstrap():
    with TemporaryDirectory(dir=".") as dirname:
        dstore = io.get_data_store(DATADIR / "3000bp.tinydb")
        reader = io.load_db()
        boostrap = create_bootstrap_app(1, discrete_edges=["758", "443154"])
        outpath = os.path.join(os.getcwd(), dirname, "tempdir.tinydb")
        writer1 = io.write_db(outpath)

        process = reader + boostrap + writer1
        process.apply_to(dstore[:1])

        loader = io.load_db()
        dstore2 = io.get_data_store(outpath)
        conv = get_convergence_bstrap(loader(dstore2[0]))
        print(conv)

        assert isinstance(conv["convergence"], float)
        assert isinstance(conv["fg_edge"], str)
        assert conv["source"] == reader(dstore[0]).info.source
        assert len(process.data_store.summary_incomplete) == 0
