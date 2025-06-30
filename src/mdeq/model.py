from cogent3 import get_moltype
from cogent3.app import evo
from cogent3.app.composable import NotCompleted, define_app
from cogent3.app.result import model_result
from cogent3.app.typing import SerialisableType
from cogent3.evolve.models import register_model
from cogent3.evolve.ns_substitution_model import GeneralStationary
from numpy import finfo

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

    # turning off selection of pade for now, possibly related to
    # cogent3 issue #993

    return evo.model(
        "GSN",
        tree=tree,
        opt_args=opt_args,
        lf_args=dict(discrete_edges=discrete_edges, expm=None),
        upper=RATE_PARAM_UPPER,
        optimise_motif_probs=True,
    )


def GN_sm(tree=None, discrete_edges=None, opt_args=None):
    opt_args = opt_args or {}
    opt_args = {"max_restarts": 5, "tolerance": 1e-8, **opt_args}

    return evo.model(
        "GN",
        tree=tree,
        opt_args=opt_args,
        lf_args=dict(discrete_edges=discrete_edges, expm=None),
        upper=RATE_PARAM_UPPER,
        optimise_motif_probs=True,
    )


# the machine precision limit
_eps = finfo(float).eps


@define_app
def mles_within_bounds(
    result: model_result,
    lower=1e-5,
    upper=RATE_PARAM_UPPER,
) -> model_result | SerialisableType:
    """validate fitted model rate parameter estimates are not close to the bounds

    Returns
    -------
    result if parameter estimate are further from the boundaries than machine
    precision epsilon, NotCompleted otherwise
    """
    if isinstance(result, NotCompleted):
        return result

    exclude_cols = {"edge", "parent", "length"}

    tables = result.lf.get_statistics()
    for table in tables:
        # if time-het model, rate params in table with "edge params" title
        # otherwise, rate params in table with "global params" title
        if table.title in ("edge params", "global params"):
            arr = table[:, [c for c in table.columns if c not in exclude_cols]].array
            if not all([(arr.min() - lower) > _eps, (upper - arr.max()) > _eps]):
                minval = arr.min()
                maxval = arr.max()
                return NotCompleted(
                    "FAIL",
                    "mles_within_bounds",
                    f"({minval:.1e}, {maxval:.1f}) params are close to bounds ({lower:.1e}, {upper:.1f})",
                    source=result,
                )
    return result
