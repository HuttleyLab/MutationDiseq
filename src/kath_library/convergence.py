from numpy.linalg import eig


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
