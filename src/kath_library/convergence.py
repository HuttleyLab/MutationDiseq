from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from numpy import array
from numpy.linalg import eig, norm
from scipy.linalg import expm

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__version__ = "2021.07.23"

def eigII(Q):
    """
    Returns the absolute value of the second largest eigenvalue from the rate matrix
    """
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


def _get_convergence(mc):

    gn = mc["mcr"]["GN"]
    fg_edge = mc["mcr"].source["fg_edge"]

    Q_darray = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)
    pi_darray = gn.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi = array([pi_darray[i] for i in Q_darray.keys()])
    Q = Q_darray.to_array()

    t = gn.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    result = generic_result(source=mc.source)
    result.update([("convergence", conv), ("fg_edge", fg_edge), ("source", mc.source)])

    return result


get_convergence = user_function(
    _get_convergence, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
