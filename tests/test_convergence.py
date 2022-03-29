import json
import pathlib

import pytest

from cogent3 import load_aligned_seqs, make_tree
from cogent3.app import io
from cogent3.maths.matrix_exponential_integration import expected_number_subs
from cogent3.util.deserialise import deserialise_object
from numpy import array, diag_indices, mean, std
from numpy.random import default_rng
from numpy.testing import assert_almost_equal

from mdeq.convergence import (
    bootstrap_to_nabla,
    convergence,
    delta_nabla,
    get_nabla,
    unit_nonstationary_Q,
    unit_stationary_Q,
)
from mdeq.model import GN_sm


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"

loader = io.load_db()


@pytest.fixture(scope="session")
def opt_args():
    """settings for faster optimisation during testing."""
    return {"max_evaluations": 10, "limit_action": "ignore", "max_restarts": 1}


@pytest.fixture(scope="session")
def pi_Q_gtr():
    pi = array(
        [
            0.2505424026859996,
            0.07241230639286972,
            0.5281794353326765,
            0.14886585558845405,
        ]
    )
    Q = array(
        [
            [
                -0.17455278757517198,
                0.058340638985947,
                0.05728142637446871,
                0.05893072221475628,
            ],
            [
                0.20185524524619958,
                -0.49216829003818047,
                0.13586228894387486,
                0.15445075584810602,
            ],
            [
                0.027171497474340816,
                0.018626438358100565,
                -0.09083073160277577,
                0.0450327957703344,
            ],
            [
                0.09918086775065285,
                0.07512895022752823,
                0.15977738177371378,
                -0.3340871997518949,
            ],
        ]
    )
    return pi, Q


@pytest.fixture()
def pi_Q():
    """pi0 and Q for a non-stationary process."""
    pi = array(
        [
            0.32299999999999995,
            0.26366666666666666,
            0.26366666666666666,
            0.14966666666666667,
        ]
    )
    Q = array(
        [
            [
                -0.1756505156896979,
                0.04785472427668278,
                0.061937684487172545,
                0.06585810692584257,
            ],
            [
                0.2556183740709909,
                -0.49677205172124006,
                0.11120873609843097,
                0.12994494155181815,
            ],
            [
                0.025005623046479422,
                0.021294409870822213,
                -0.09273954148445357,
                0.04643950856715194,
            ],
            [
                0.08460220390537197,
                0.08443523025341788,
                0.1571661576846348,
                -0.3262035918434247,
            ],
        ]
    )

    return pi, Q


def test_convergence_non_zero(pi_Q):
    """Convergence of a non-stationary process should be greater than 0."""
    conv = convergence(*pi_Q, 0.17728507678400238)
    assert conv >= 0


def test_convergence_GTR(pi_Q_gtr):
    """convergence of a stationary process should be 0."""
    pi, Q = pi_Q_gtr
    t = 0.17708101717074656
    # it's a stationary process
    assert_almost_equal(pi @ Q, 0.0)

    conv = convergence(pi, Q, t)
    assert_almost_equal(conv, 0, decimal=10)


# testing delta_nabla dataclass
def test_make_delta_nabla():
    """works if list, tuple or numpy array used."""
    data = [0.2, 1, 1.8]
    fg_edge = "blah"
    for _type_ in (tuple, list, array):
        obj = delta_nabla(3.0, _type_(data), fg_edge)
        assert obj.mean_null == 1
        assert obj.size_null == 3


def test_fail_make_delta_nabla():
    """fail if empty null distribution."""
    for _type_ in (tuple, list, array):
        for data in ([], [2]):
            with pytest.raises(ValueError):
                delta_nabla(3.0, _type_(data), "blah", 3)


def test_delta_nabla_value():
    """given statistic and null of statistic computes correct delta_nabla."""
    rng = default_rng()
    size = 67
    obs_nabla = rng.uniform(low=1e-6, high=100, size=1)[0]
    null_nabla = rng.uniform(low=1e-6, high=100, size=size)
    dnab = delta_nabla(obs_nabla, null_nabla, "blah")
    assert dnab.obs_nabla == obs_nabla
    assert dnab.size_null == size
    assert dnab.fg_edge == "blah"
    mean_null = mean(null_nabla)
    assert mean_null == dnab.mean_null
    std_null = std(null_nabla, ddof=1)
    assert std_null == dnab.std_null

    assert dnab.delta_nabla == obs_nabla - mean_null


def test_rich_dict():
    """dict can be json formatted."""
    obj = delta_nabla(3.0, (0.2, 1.0, 1.8), "blah")
    got = obj.to_rich_dict()
    _ = json.dumps(got)  # should not fail
    # type value has correct name
    assert got["type"].endswith(delta_nabla.__name__)


def test_roundtrip_json():
    """direct deserialisation works."""
    obj = delta_nabla(3.0, (0.2, 1.0, 1.8), "blah")
    j = obj.to_json()
    g = delta_nabla.from_dict(json.loads(j))
    assert g == obj


def test_cogent3_deserialisation():
    """works with deserialise_object."""
    obj = delta_nabla(3.0, (0.2, 1.0, 1.8), "blah")
    j = obj.to_json()
    g = deserialise_object(j)
    assert isinstance(g, delta_nabla)
    assert g == obj


def test_get_nabla(toe_bstrap):
    """correctly computes nabla stats."""
    result = toe_bstrap[0]
    obs_result = result["observed"]["GN"]
    fg_edge, n = get_nabla(None, obs_result)
    assert fg_edge in obs_result.lf.tree.get_tip_names()
    assert isinstance(n, float)
    with pytest.raises(AssertionError):
        get_nabla(None, gn_result=obs_result, time_delta=obs_result)


@pytest.fixture(scope="session")
def alignment_tree():
    aln = load_aligned_seqs(DATADIR / "brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["Human", "Mouse", "Rhesus", "Wombat"]).no_degenerates(
        motif_length=3
    )
    tree = make_tree("(Wombat,Mouse,(Human,Rhesus))")
    return aln, tree


def test_get_nabla_mixes(alignment_tree, opt_args):
    """should raise a NotImplementedError if number of Q != number edges or
    1."""
    aln, tree = alignment_tree

    mod = GN_sm(tree=tree, discrete_edges=["Wombat"], opt_args=opt_args)
    result = mod(aln)
    # too many Q
    with pytest.raises(NotImplementedError):
        get_nabla(result)

    # but works for all Q
    mod = GN_sm(tree=tree, opt_args=opt_args)
    result = mod(aln)
    fg_edge, nabla = get_nabla(None, result)
    assert isinstance(nabla, float)
    assert fg_edge in aln.names


@pytest.fixture(scope="session")
def toe_bstrap():
    """tinydb with bootstrap results."""
    inpath = DATADIR / "toe-300bp.tinydb"
    dstore = io.get_data_store(inpath)
    loader = io.load_db()
    return [loader(m) for m in dstore]


def test_load_delta_nabla(toe_bstrap):
    """returns a series of delta_nabla instances."""
    app = bootstrap_to_nabla()
    results = [app(r) for r in toe_bstrap]
    assert {type(r) for r in results} == {delta_nabla}
    assert all(v.delta_nabla >= 0 for v in results)


def test_unit_ens(pi_Q):
    pi, Q = pi_Q
    result = unit_nonstationary_Q(pi, Q)
    ens = expected_number_subs(pi, result, 1.0)
    assert_almost_equal(ens, 1.0)


def test_unit_ens_gtr(pi_Q_gtr):
    """result is same as standard calibration for GTR matrix."""
    pi, Q = pi_Q_gtr
    # calibrated assuming non-stationary
    ens_Q = unit_nonstationary_Q(pi, Q)
    # calibrated assuming stationary
    tau_Q = unit_stationary_Q(pi, Q)
    assert_almost_equal(ens_Q, tau_Q)
    # time == 1
    indices = diag_indices(4)
    assert_almost_equal(-sum(pi * Q[indices]), 1.0)
