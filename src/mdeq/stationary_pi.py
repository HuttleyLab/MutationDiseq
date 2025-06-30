from numpy import allclose
from numpy.linalg import eig, inv, norm
from numpy.ma import dot as ma_dot
from numpy.ma.core import array, diag

from mdeq.numeric import (
    dot,
    sum,
    valid_probability_vector,
    valid_stochastic_matrix,
)


class OscillatingPiException(Exception):
    """Did not converge to a unique stationary distribution."""


def get_stat_pi_via_eigen(P, check_precision=True):
    """This code was provided by Gavin Huttley Obtain stationary distribution
    via EigenValue decomposition."""
    P = array(P).T
    eva, eve = eig(P)

    if check_precision:
        i = inv(eve)
        r = ma_dot(ma_dot(eve, diag(eva)), i).real
        if not allclose(P, r):
            raise ArithmeticError

    evect = eve[:, eva.round(10) == 1]
    stat_pi = evect / sum(evect.real)
    return stat_pi.flatten().real


def get_stat_pi_via_brute(
    P,
    pi_zero,
    limit_action="raise",
    max_iterations=100000,
    threshold=1e-8,
):
    """Obtain the stationary distribution via brute force."""

    delta = 10
    num_iterations = 0

    if not (valid_stochastic_matrix(P) and valid_probability_vector(pi_zero)):
        raise ValueError("Invalid psub matrix or pi vector input")

    while delta > threshold:
        new_pi = dot(pi_zero, P)
        delta = norm(abs(pi_zero - new_pi))
        pi_zero = new_pi

        if num_iterations == max_iterations:
            break

        num_iterations += 1

    if num_iterations == max_iterations and limit_action == "raise":
        raise OscillatingPiException("Reached maximum iterations without convergence")

    return new_pi
