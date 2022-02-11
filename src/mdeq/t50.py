import numpy as np

from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from cogent3.maths.matrix_exponential_integration import expected_number_subs
from cogent3.maths.measure import jsm
from cogent3.maths.optimisers import minimise
from scipy.linalg import expm

from mdeq.stationary_pi import get_stat_pi_via_brute, get_stat_pi_via_eigen
from mdeq.utils.utils import get_foreground, get_pi_0, get_pi_tip


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


class T50:
    """
    takes a rate matrix and a starting probability vector, and computes
    the expected number of substitution to the half way distribution
    """

    def __init__(self, Q, pi_0, func=jsm):
        """
        Parameters
        ----------
        Q
            a valid uncalibrated rate matrix
        pi_0
            a valid probability vector representing the initial state freqs
        func
            a callback function that takes two probability vectors (pi_zero, pi_stationary)
            and returns a "distance". Defaults to Jensen-Shannon metric
        """
        self.Q = Q
        self.pi_0 = pi_0
        self.pi_inf = self.get_stat_pi()
        self.dist_halfway = func(self.pi_0, self.pi_inf) / 2
        self.tau = 1
        self.dist_func = func

    def get_stat_pi(self):
        return get_stat_pi_via_brute(expm(self.Q), self.pi_0)

    def estimate_t50(self):
        ens_curr = expected_number_subs(self.pi_0, self.Q, 1)
        self.tau = minimise(
            self,
            xinit=self.tau,
            bounds=([1], [1e10]),
            local=True,
            show_progress=False,
            tolerance=1e-8,
        )
        ens_50 = expected_number_subs(self.pi_0, self.Q, self.tau)
        return ens_50 - ens_curr

    def distance_from_pi_zero(self, pi):
        return self.dist_func(self.pi_0, pi)

    def __call__(self, tau):
        pi_tau = dot(self.pi_0, expm(self.Q * tau))
        dist1 = self.dist_func(self.pi_0, pi_tau)
        dist2 = self.dist_func(pi_tau, self.pi_inf)
        return abs(dist1 - dist2) ** 2


def _get_t50_mc(mc):
    """
    Wrapper function to return T50 estimate from a model collection that includes a GN fit.
    Returns a generic_result
    """

    gn = mc["mcr"]["GN"]

    bg_edges = gn.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(gn.alignment.names)

    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_tip(gn, fg_edge)

    t50 = T50(Q, pi)
    t50_val = t50.estimate_t50()

    result = generic_result(source=mc.source)
    result.update([("T50", t50_val), ("fg_edge", fg_edge), ("source", mc.source)])

    return result


get_t50_mc = user_function(
    _get_t50_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_t50(gn_sm):
    """
    Wrapper function to return T50 estimate from a GN fit.
    Returns a generic_result
    """

    bg_edges = gn_sm.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(gn_sm.alignment.names)

    Q = gn_sm.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_tip(gn_sm, fg_edge)

    t50 = T50(Q, pi)
    t50_val = t50.estimate_t50()

    result = generic_result(source=gn_sm.source)

    result.update([("T50", t50_val)])
    return result


get_t50 = user_function(
    _get_t50, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_t50_bstrap(result):
    """
    Wrapper function to return convergence estimate from a generic_result generated from a bootstrap app.
    Returns a generic_result
    """

    gn = result["observed"].alt

    bg_edges = gn.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(gn.alignment.names)

    Q = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_tip(gn, fg_edge)

    t50 = T50(Q, pi)
    t50_val = t50.estimate_t50()

    result = generic_result(source=result.source)
    result.update([("T50", t50_val), ("fg_edge", fg_edge), ("source", result.source)])

    return result


get_t50_bstrap = user_function(
    _get_t50_bstrap, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
