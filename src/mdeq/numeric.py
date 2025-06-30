import math

import numpy

try:
    from accupy import fdot as dot
    from accupy import fsum as sum
except ImportError:
    sum = math.fsum
    dot = numpy.dot

from cogent3.util.dict_array import DictArray


def valid_stochastic_matrix(matrix):
    """returns True if all rows sum to 1 and all entries are valid
    probabilities."""
    if isinstance(matrix, DictArray):
        matrix = matrix.to_array()

    row_sum_one = numpy.allclose(
        [sum(row) for row in matrix],
        numpy.ones(len(matrix)),
        rtol=1e-10,
        atol=1e-14,
    )
    all_in_unit_interval = all(0 <= i <= 1 for i in numpy.nditer(matrix))

    return row_sum_one and all_in_unit_interval


def valid_probability_vector(vector):
    """returns True if vector sums to 1 and all entries are valid
    probabilities."""
    if isinstance(vector, DictArray):
        vector = vector.to_array()
    row_sum_one = math.isclose(sum(vector), 1, rel_tol=1e-10, abs_tol=1e-14)
    all_in_unit_interval = all(0 <= i <= 1 for i in vector)

    return row_sum_one and all_in_unit_interval


def valid_rate_matrix(matrix):
    """returns True if off-diagonal elements are positive and row sums are
    0."""
    if isinstance(matrix, DictArray):
        matrix = matrix.to_array()
    row_sum_zero = numpy.allclose(
        [sum(row) for row in matrix],
        numpy.zeros(len(matrix)),
        rtol=1e-10,
        atol=1e-14,
    )

    off_diagonal_elem = [
        elem for i, row in enumerate(matrix) for j, elem in enumerate(row) if i != j
    ]

    off_diagonal_elem_positive = all(x > 0 for x in off_diagonal_elem)
    return row_sum_zero and off_diagonal_elem_positive


def fix_rounding_error(x, round_error=1e-14):
    """If x is almost in the range 0, fixes it.

    Specifically, if x is between -round_error and 0, returns 0.
    """
    return 0 if -round_error < x < 0 else x
