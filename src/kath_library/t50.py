from accupy import fdot as dot
from cogent3.maths.matrix_exponential_integration import expected_number_subs
from cogent3.maths.measure import jsm
from cogent3.maths.optimisers import minimise
from scipy.linalg import expm

from kath_library.stationary_pi import get_stat_pi_via_brute

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]
__version__ = "2021.07.01"
__maintainer__ = "Katherine Caley"
__email__ = "katherine.caley@anu.edu.au"
__status__ = "develop"


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
            a valid rate matrix
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
        ens0 = expected_number_subs(self.pi_0, self.Q, self.tau)
        self.tau = minimise(
            self,
            xinit=self.tau,
            bounds=([0], [1e10]),
            local=True,
            show_progress=False,
            tolerance=1e-9,
        )
        ens_total = expected_number_subs(self.pi_0, self.Q, self.tau)
        return ens_total - ens0

    def distance_from_pi_zero(self, pi):
        return self.dist_func(self.pi_0, pi)

    def __call__(self, tau):
        pi_tau = dot(self.pi_0, expm(self.Q * tau))
        dist = self.dist_func(self.pi_0, pi_tau)
        return abs(self.dist_halfway - dist) ** 2
