import math

import numpy as np
from cogent3 import make_table
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from cogent3.maths.stats import chisqprob
from cogent3.util.dict_array import DictArray, DictArrayTemplate
from scipy.linalg import expm

from kath_library.stationary_pi import get_stat_pi_via_brute
from kath_library.utils.utils import get_foreground


def stationarity_indices(model_result):
    """
    model_result
        GN instance
    return
        [STI1, STI2, STI3] corresponding to the set of stationarity indices presented in Squartini and Arndt, 2008
    """

    fg_edge = get_foreground(model_result.alignment)

    pi_0 = model_result.lf.get_motif_probs()
    Q = model_result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)

    assert pi_0.keys() == Q.keys()
    nt_order = Q.keys()

    pi_inf_array = get_stat_pi_via_brute(expm(Q.to_array()), pi_0.to_array())
    template = DictArrayTemplate(nt_order)
    pi_inf = template.wrap(pi_inf_array)

    STI1 = (pi_0["C"] - pi_inf["C"]) + (pi_0["G"] - pi_inf["G"])
    STI2 = (pi_0["A"] - pi_inf["A"]) - (pi_0["T"] - pi_inf["T"])
    STI3 = (pi_0["C"] - pi_inf["C"]) - (pi_0["G"] - pi_inf["G"])

    return [STI1, STI2, STI3]


def chi_squared_test(model_result):
    """
    model_result
        GN instance
    return
        table containing the results of a chi^2 test of difference in nucleotide composition,
        corresponding to the test for significance of non-zero STIs presented in Squartini and Arndt, 2008
    """

    fg_edge = get_foreground(model_result.alignment)

    pi_0 = model_result.lf.get_motif_probs()
    Q = model_result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)

    assert pi_0.keys() == Q.keys()
    nt_order = Q.keys()

    pi_inf_array = get_stat_pi_via_brute(expm(Q.to_array()), pi_0.to_array())
    template = DictArrayTemplate(nt_order)
    pi_inf = template.wrap(pi_inf_array)

    chi_counter = float(0.0)
    for nt in nt_order:
        chi_counter += math.pow((pi_0[nt] - pi_inf[nt]), 2) / pi_inf[nt]

    chi_2 = chi_counter * len(model_result.alignment)

    table = make_table(
        header=["chi_2", "df", "p"],
        rows=[[chi_2, 3, chisqprob(chi_2, 3)]],
        digits=2,
        space=3,
    )

    return table


def _get_STIs_mc(mc):
    """
    Wrapper function to return STI estimate from a model collection that includes a GN fit.
    Returns a generic_result
    """

    gn = mc["mcr"]["GN"]
    STIs = stationarity_indices(gn)

    result = generic_result(source=mc.source)
    result.update(
        [
            ("STIs", STIs),
            ("STI1", STIs[0]),
            ("STI2", STIs[1]),
            ("STI3", STIs[2]),
            ("source", mc.source),
        ]
    )

    return result


get_STIs_mc = user_function(
    _get_STIs_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_chi2_mc(mc):
    """
    Wrapper function to return STI estimate from a model collection that includes a GN fit.
    Returns a generic_result
    """

    gn = mc["mcr"]["GN"]
    chi_2 = chi_squared_test(gn)

    result = generic_result(source=mc.source)
    result.update([("chi_2", chi_2), ("source", mc.source)])

    return result


get_chi2_mc = user_function(
    _get_chi2_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
