from cogent3 import get_moltype
from cogent3.app import evo
from cogent3.evolve.models import register_model
from cogent3.evolve.ns_substitution_model import GeneralStationary


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


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
    )
