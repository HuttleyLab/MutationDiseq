import numpy as np
from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from cogent3.maths.matrix_exponential_integration import expected_number_subs
from cogent3.maths.measure import jsm
from cogent3.maths.optimisers import minimise
from scipy.linalg import expm

from kath_library.stationary_pi import get_stat_pi_via_brute, get_stat_pi_via_eigen

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
            bounds=([0], [1e10]),
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


def _get_t50(mc):

    gn = mc["mcr"]["GN"]
    fg_edge = mc["mcr"].source["fg_edge"]

    Q_darray = gn.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)
    pi_darray = gn.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi = np.array([pi_darray[i] for i in Q_darray.keys()])
    Q = Q_darray.to_array()

    t50 = T50(Q, pi)
    t50_val = t50.estimate_t50()

    result = generic_result(source=mc.source)
    result.update([("t50", t50_val), ("fg_edge", fg_edge), ("source", mc.source)])

    return result


get_t50 = user_function(
    _get_t50, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
