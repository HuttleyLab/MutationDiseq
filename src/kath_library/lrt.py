from cogent3 import get_moltype
from cogent3.app import evo, io
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result

from kath_library.model import GS_instance

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


def hypothesis(mc):
    hyp = mc["mcr"].get_hypothesis_result("GS", "GN")
    hyp.source = hyp.source["source"]
    return hyp


get_lrt = user_function(
    hypothesis, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def get_no_init_model_coll(aln):
    """
    creates a model_collection object without sequential fitting to be called with given alignment.

    Parameters
    ----------
    aln : alignment to fit models to. NOTE! aln needs the foreground edge as an entry to the .info dictionary!

    Returns
    -------
    model_collection_result containing GS and GN models (without sequential fitting)
    """

    try:
        fg_edge = aln.info.fg_edge
    except AttributeError:
        raise AttributeError("Alignment needs a info.fg_edge attribute")

    if fg_edge is None:
        raise AttributeError("Alignment needs a info.fg_edge attribute")

    bg_edges = list({fg_edge} ^ set(aln.names))

    GS = evo.model(
        GS_instance(),
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )
    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )
    mc = evo.model_collection(GS, GN, sequential=False)

    mc_result = mc(aln)

    return mc_result


def get_init_model_coll(aln):
    """
    creates a model_collection object with sequential fitting to be called with given aln.

    Parameters
    ----------
    aln : alignment to fit models to. NOTE! aln needs the foreground edge as an entry to the .info dictionary!

    Returns
    -------
    model_collection_result containing GTR, GS and GN models (with sequential fitting)
    """

    try:
        fg_edge = aln.info.fg_edge
    except AttributeError:
        raise AttributeError("Alignment needs a info.fg_edge attribute")

    if fg_edge is None:
        raise AttributeError("Alignment needs a info.fg_edge attribute")

    bg_edges = list({fg_edge} ^ set(aln.names))

    GTR = evo.model(
        "GTR",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )
    GS = evo.model(
        GS_instance(),
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )
    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )
    mc = evo.model_collection(GTR, GS, GN, sequential=True)

    mc_result = mc(aln)

    return mc_result


def _get_no_init_hypothesis(aln):
    mc_result = get_no_init_model_coll(aln)
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result


get_no_init_hypothesis = user_function(
    _get_no_init_hypothesis,
    input_types=SERIALISABLE_TYPE,
    output_types=SERIALISABLE_TYPE,
)


def _get_init_hypothesis(aln):
    mc_result = get_init_model_coll(aln)
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result


get_init_hypothesis = user_function(
    _get_init_hypothesis, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
