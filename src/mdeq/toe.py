from cogent3.app import evo
from cogent3.app.composable import SERIALISABLE_TYPE, NotCompleted, appify
from cogent3.app.result import generic_result
from cogent3.util.misc import extend_docstring_from

from mdeq.model import RATE_PARAM_UPPER
from mdeq.utils import get_foreground


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


NULL_TOE = "GSN"
ALT_TOE = "GN"


@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_lrt(mc):
    if isinstance(mc, NotCompleted):
        return mc

    hyp = mc["mcr"].get_hypothesis_result("GSN", "GN")
    hyp.source = hyp.source["source"]
    return hyp


def toe_on_edge(aln, tree=None, with_gtr=False, sequential=False, opt_args=None):
    """make app to test for equilibrium with a dynamically defined background
    edge.

    Parameters
    ----------
    aln : Alignment
        must have a fg_edge value in the dict to identify the sequence on the
        foreground edge
    tree
        phylogenetic tree
    with_gtr : bool
        use GTR to get initial estimates for GSN
    sequential : bool
        uses MLEs from nested models as initial values
    opt_args : dict
        dict specifying arguments to likelihood function optimisation.
        Overrides internal settings.

    Notes
    -----
    The fg_edge value is a name in the alignment and can differ between
    alignments. The other edges are modelled using a discrete-time Markov
    process. We do not advise doing this for > 3 edges.

    Defaults to a strictly continuous-time process if no such fg_edge can
    be inferred, i.e. null is a single GSN process, alt is a single GN
    process.

    Returns
    -------
    model_collection
    """
    model_names = ["GTR"] if with_gtr else []
    model_names.extend([NULL_TOE, ALT_TOE])

    fg_edge = get_foreground(aln)
    if fg_edge is None:
        raise ValueError(f"alignment.info {aln.info!r} missing 'fg_edge' value")

    bg_edges = list({fg_edge} ^ set(aln.names)) if fg_edge else None
    opt_args = opt_args or {}
    opt_args = {"max_restarts": 5, "tolerance": 1e-8, **opt_args}
    # turning off selection of pade for now, possibly related to
    # cogent3 issue #993
    lf_args = dict(discrete_edges=bg_edges, expm=None)
    models = [
        evo.model(
            mn,
            tree=tree,
            opt_args=opt_args,
            lf_args=lf_args,
            time_het=dict(upper=RATE_PARAM_UPPER),
            optimise_motif_probs=True,
        )
        for mn in model_names
    ]
    return evo.model_collection(*models, sequential=sequential)


def get_no_init_model_coll(aln, opt_args=None):
    """fits GSN and GN **without** sequential fitting

    Parameters
    ----------
    aln
        alignment to fit models to.
    opt_args : dict
        settings passed to the optimiser

    Notes
    -----
    aln needs the foreground edge as an entry to the .info dictionary!

    Returns
    -------
    model_collection_result containing GS and GN models (without sequential fitting)
    """
    return toe_on_edge(aln, with_gtr=False, sequential=False, opt_args=opt_args)(aln)


@extend_docstring_from(get_no_init_model_coll)
def get_init_model_coll(aln, opt_args=None):
    """fits GTR, GSN, GN models **with** sequential fitting"""
    return toe_on_edge(aln, with_gtr=True, sequential=True, opt_args=opt_args)(aln)


@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_no_init_hypothesis(aln, opt_args=None):
    if isinstance(aln, NotCompleted):
        return aln

    mc_result = toe_on_edge(aln, with_gtr=False, sequential=False, opt_args=opt_args)(
        aln
    )
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result


@extend_docstring_from(toe_on_edge)
@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_init_hypothesis(aln, opt_args=None):
    if isinstance(aln, NotCompleted):
        return aln

    mc_result = toe_on_edge(aln, with_gtr=True, sequential=True, opt_args=opt_args)(aln)
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result
