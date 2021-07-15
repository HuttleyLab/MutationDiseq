from numpy.linalg import eig

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__version__ = "2021.07.07"


def convergence(Q):
    """
    Returns the absolute value of the second largest eigenvalue from the rate matrix
    """
    v, r = eig(Q)
    v = v.flatten()
    v.sort()
    eigII = v[-2]
    conv = abs(eigII)

    return conv
