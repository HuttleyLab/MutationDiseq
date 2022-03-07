from cogent3.app import evo
from cogent3.app.composable import SERIALISABLE_TYPE, appify
from cogent3.app.result import generic_result
from cogent3.util.misc import extend_docstring_from

from mdeq.model import GSN
from mdeq.utils.utils import get_foreground


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]


@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_lrt(mc):
    hyp = mc["mcr"].get_hypothesis_result("GSN", "GN")
    hyp.source = hyp.source["source"]
    return hyp


def toe_on_edge(aln, with_gtr=False, sequential=False, opt_args=None):
    """make app to test for equilibrium with a dynamically defined background edge.

    Parameters
    ----------
    aln : Alignment
        must have a fg_edge value in the dict to identify the sequence on the
        foreground edge
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

    Returns
    -------
    model_collection
    """
    model_names = ["GTR"] if with_gtr else []
    model_names.extend(["GSN", "GN"])

    fg_edge = get_foreground(aln)
    bg_edges = list({fg_edge} ^ set(aln.names)) if fg_edge else None
    sm_args = dict(optimise_motif_probs=True)
    opt_args = opt_args or {}
    opt_args = dict(max_restarts=5, tolerance=1e-8, **opt_args)
    lf_args = dict(discrete_edges=bg_edges, expm="pade")

    models = [
        evo.model(mn, sm_args=sm_args, opt_args=opt_args, lf_args=lf_args)
        for mn in model_names
    ]
    return evo.model_collection(*models, sequential=sequential)


def get_no_init_model_coll(aln, opt_args=None):
    """creates a model_collection object without sequential fitting to be
    called with given alignment.

    Parameters
    ----------
    aln : alignment to fit models to. NOTE! aln needs the foreground edge as an entry to the .info dictionary!

    Returns
    -------
    model_collection_result containing GS and GN models (without sequential fitting)
    """
    return toe_on_edge(aln, with_gtr=False, sequential=False, opt_args=opt_args)(aln)


def get_init_model_coll(aln, opt_args=None):
    """creates a model_collection object with sequential fitting to be called
    with given aln.

    Parameters
    ----------
    aln : alignment to fit models to. NOTE! aln needs the foreground edge as an entry to the .info dictionary!

    Returns
    -------
    model_collection_result containing GTR, GS and GN models (with sequential fitting)
    """
    return toe_on_edge(aln, with_gtr=True, sequential=True, opt_args=opt_args)(aln)


@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_no_init_hypothesis(aln, opt_args=None):
    mc_result = toe_on_edge(aln, with_gtr=False, sequential=False, opt_args=opt_args)(
        aln
    )
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result


@extend_docstring_from(toe_on_edge)
@appify(input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE)
def get_init_hypothesis(aln, opt_args=None):
    mc_result = toe_on_edge(aln, with_gtr=True, sequential=True, opt_args=opt_args)(aln)
    result = generic_result(source=aln.info.source)
    result.update([("mcr", mc_result)])
    return result
