from cogent3 import get_moltype
from cogent3.app import evo
from cogent3.app.composable import SERIALISABLE_TYPE, NotCompleted, appify
from cogent3.evolve.models import register_model
from cogent3.evolve.ns_substitution_model import GeneralStationary
from numpy import allclose


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

RATE_PARAM_UPPER = 100


@register_model("nucleotide")
def GSN(**kwargs):
    """A General Stationary Nucleotide substitution model instance."""
    kwargs["optimise_motif_probs"] = kwargs.get("optimise_motif_probs", True)
    kwargs["name"] = kwargs.get("name", "GSN")
    return GeneralStationary(get_moltype("dna").alphabet, **kwargs)


def GS_sm(tree=None, discrete_edges=None, opt_args=None):
    opt_args = opt_args or {}
    opt_args = {"max_restarts": 5, "tolerance": 1e-8, **opt_args}
    return evo.model(
        GSN(),
        tree=tree,
        sm_args=dict(optimise_motif_probs=True),
        opt_args=opt_args,
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
        upper=RATE_PARAM_UPPER,
    )


def GN_sm(tree=None, discrete_edges=None, opt_args=None):
    opt_args = opt_args or {}
    opt_args = {"max_restarts": 5, "tolerance": 1e-8, **opt_args}

    return evo.model(
        "GN",
        tree=tree,
        sm_args=dict(optimise_motif_probs=True),
        opt_args=opt_args,
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
        upper=RATE_PARAM_UPPER,
    )


@appify(SERIALISABLE_TYPE, SERIALISABLE_TYPE)
def mles_near_bounds(result, lower=1e-5, upper=RATE_PARAM_UPPER):
    """
    Returns a NotCompleted if parameter estimates of model_result are at the bounds, else returns model_result.

    """
    tables = result.lf.get_statistics()
    for table in tables:
        # if time-het model, rate params in table with "edge params" title
        # otherwise, rate params in table with "global params" title
        if table.title in ("edge params", "global params"):
            arr = table[:, [c for c in table.columns if c != "length"]].array
            if not all([(arr.min() - lower) > 0.0, (upper - arr.max()) > 0.0]):
                minval = arr.min()
                maxval = arr.max()
                return NotCompleted(
                    "FAIL",
                    "params_near_bounds",
                    f"({minval:.1e}, {{maxval:.1f}}) params not within bounds ({lower:.1e}, {upper:.1f})",
                    source=str(result.source),
                )
    return result
