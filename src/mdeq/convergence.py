from accupy import fdot as dot
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from numpy import array, std
from numpy.linalg import eig, norm
from scipy.linalg import expm

from mdeq.model import GN_sm, GS_sm
from mdeq.utils.utils import get_foreground, get_pi_0, get_pi_tip

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Ben Kaehler"]


def eigII(Q):
    """
    returns the dominant eigenvalue
    """
    v, r = eig(Q)
    v = v.flatten()
    v.sort()
    eigII = v[-2]
    if isinstance(eigII, complex):
        eigII = eigII.real

    return eigII


def convergence(pi_0, Q, t):
    """
    a measure of how fast pi(t) is changing
    """
    pi_deriv = dot(pi_0, dot(Q, expm(Q * t)))
    conv = norm(pi_deriv)

    return conv


def _get_convergence_mc(mc):
    """
    Wrapper function to return convergence estimate from a model collection that includes a GN fit.
    Returns a generic_result
    """

    alt = mc["mcr"]["GN"]

    bg_edges = alt.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(alt.alignment.names)

    Q = alt.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(alt)
    t = alt.lf.get_param_value("length", edge=fg_edge)

    observed_conv = convergence(pi, Q, t)

    null = mc["mcr"]["GSN"]
    GN = GN_sm(bg_edges)
    neutral_convs = []

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (mc.source, i)
        sim_aln.info.fg_edge = fg_edge

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (observed_conv - neutral_mean) / neutral_std

    result = generic_result(source=mc.source)
    result.update(
        [
            ("convergence_normalised", conv_normalised),
            ("convergence", observed_conv),
            ("fg_edge", fg_edge),
            ("source", mc.source),
            ("model_fit", alt),
        ]
    )

    return result


get_convergence_mc = user_function(
    _get_convergence_mc, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_convergence(gn_sm):
    """
    Wrapper function to return convergence estimate from a non-stationary model fit.
    Returns a generic_result
    """

    bg_edges = gn_sm.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(gn_sm.alignment.names)

    Q = gn_sm.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(gn_sm)
    t = gn_sm.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    GS = GS_sm(bg_edges)
    GN = GN_sm(bg_edges)

    null = GN(gn_sm.alignment)
    neutral_convs = []

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "simalign %d" % (i)
        sim_aln.info.fg_edge = fg_edge

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (conv - neutral_mean) / neutral_std

    result = generic_result(source=gn_sm.source)
    result.update([("convergence", conv), ("convergence_normalised", conv_normalised)])

    return result


get_convergence = user_function(
    _get_convergence, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _get_convergence_bstrap(result):
    """
    Wrapper function to return convergence estimate from a generic_result generated from a bootstrap app..
    Returns a generic_result
    """

    alt = result["observed"].alt
    null = result["observed"].null

    bg_edges = alt.lf.to_rich_dict()["likelihood_construction"]["discrete_edges"]
    (fg_edge,) = set(bg_edges) ^ set(alt.alignment.names)

    Q = alt.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False).to_array()
    pi = get_pi_0(alt)
    t = alt.lf.get_param_value("length", edge=fg_edge)

    conv = convergence(pi, Q, t)

    GN = GN_sm(bg_edges)
    neutral_convs = []

    while len(neutral_convs) < 10:
        i = len(neutral_convs) + 1
        sim_aln = null.lf.simulate_alignment()
        sim_aln.info.source = "%s - simalign %d" % (result.source, i)
        sim_aln.info.fg_edge = fg_edge

        try:
            sim_model_fit = GN(sim_aln)
            sim_Q = sim_model_fit.lf.get_rate_matrix_for_edge(
                fg_edge, calibrated=False
            ).to_array()
            sim_pi = get_pi_0(sim_model_fit)
            sim_t = sim_model_fit.lf.get_param_value("length", edge=fg_edge)

            sim_conv = convergence(sim_pi, sim_Q, sim_t)
            neutral_convs.append(sim_conv)
        except ValueError:
            sim_result = None

    neutral_mean = array(neutral_convs).mean()
    neutral_std = std(array(neutral_convs), ddof=1)

    conv_normalised = (conv - neutral_mean) / neutral_std

    result = generic_result(source=result.source)
    result.update(
        [
            ("convergence_normalised", conv_normalised),
            ("convergence", conv),
            ("fg_edge", fg_edge),
            ("source", result.source),
        ]
    )

    return result


get_convergence_bstrap = user_function(
    _get_convergence_bstrap,
    input_types=SERIALISABLE_TYPE,
    output_types=SERIALISABLE_TYPE,
)
