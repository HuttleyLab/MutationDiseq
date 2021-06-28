from numpy.linalg import eig

def get_eig_ii(Q):
    """
    Returns the second largest eigenvalue from the rate matrix
    """
    v, r = eig(Q)
    v = v.flatten()
    v.sort()
    eigII = v[-2]

    return eigII
