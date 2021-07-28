from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from numpy import array
from numpy.linalg import eig, norm
from scipy.linalg import expm

from kath_library.utils.utils import get_pi_0, get_pi_tip

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


def eigII(Q):
    v, r = eig(Q)
    v = v.flatten()
    v.sort()
    eigII = v[-2]
    if isinstance(eigII, complex):
        eigII = eigII.real

    return eigII


def convergence(pi_0, Q, t):
    """
    a measure of how fast pi(t) is changing
    """
    pi_deriv = dot(pi_0, dot(Q, expm(Q * t)))
    conv = norm(pi_deriv)

    return conv


def _get_convergence_mc(mc):
    """
    Wrapper function to return convergence estimate from a model collection that includes a GN fit.
    Returns a generic_result
    """

    gn = mc["mcr"]["GN"]
    fg_edge = mc["mcr"].source["fg_edge"]

    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_tip(gn, fg_edge)
    t = gn.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    result = generic_result(source=mc.source)
    result.update([("convergence", conv), ("fg_edge", fg_edge), ("source", mc.source)])

    return result


get_convergence_mc = user_function(
    _get_convergence_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_convergence(gn_sm):
    """
    Wrapper function to return convergence estimate from a GN fit.
    Returns a generic_result
    """

    fg_edge = gn_sm.alignment.info.fg_edge

    Q = gn_sm.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_tip(gn_sm, fg_edge)
    t = gn_sm.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    result = generic_result(source=gn_sm.source)
    result.update([("convergence", conv)])

    return result


get_convergence = user_function(
    _get_convergence, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
