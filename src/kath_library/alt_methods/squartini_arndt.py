import math

import numpy as np
from cogent3 import make_table
from cogent3.maths.stats import chisqprob
from cogent3.util.dict_array import DictArray, DictArrayTemplate
from scipy.linalg import expm

from kath_library.stationary_pi import get_stat_pi_via_brute
from kath_library.utils.utils import get_foreground, get_pi_tip


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

    pi_tip_unordered = model_result.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi_tip_array = np.array([pi_tip_unordered[i] for i in Q.keys()])
    pi_inf_array = get_stat_pi_via_brute(expm(Q.to_array()), np.array(pi_tip_array))

    template = DictArrayTemplate(nt_order)

    pi_tip = template.wrap(pi_tip_array)
    pi_inf = template.wrap(pi_inf_array)

    STI1 = (pi_tip["C"] - pi_inf["C"]) + (pi_tip["G"] - pi_inf["G"])
    STI2 = (pi_tip["A"] - pi_inf["A"]) - (pi_tip["T"] - pi_inf["T"])
    STI3 = (pi_tip["C"] - pi_inf["C"]) - (pi_tip["G"] - pi_inf["G"])

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

    pi_tip_unordered = model_result.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi_tip_array = np.array([pi_tip_unordered[i] for i in Q.keys()])
    pi_inf_array = get_stat_pi_via_brute(expm(Q.to_array()), np.array(pi_tip_array))

    template = DictArrayTemplate(nt_order)

    pi_tip = template.wrap(pi_tip_array)
    pi_inf = template.wrap(pi_inf_array)

    chi_counter = float(0.0)
    for nt in nt_order:
        chi_counter += math.pow((pi_tip[nt] - pi_inf[nt]), 2) / pi_inf[nt]

    chi_2 = chi_counter * len(model_result.alignment)

    table = make_table(
        header=["chi_2", "df", "p"],
        rows=[[chi_2, 3, chisqprob(chi_2, 3)]],
        digits=2,
        space=3,
    )

    return table
