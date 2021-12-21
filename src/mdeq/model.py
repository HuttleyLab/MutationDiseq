from cogent3 import get_moltype
from cogent3.app import evo
from cogent3.evolve.ns_substitution_model import GeneralStationary


def GS_instance():
    """
    A General Stationary Nucleotide substitution model instance.
    """
    return GeneralStationary(
        get_moltype("dna").alphabet, optimise_motif_probs=True, name="GS"
    )


def GS_sm(discrete_edges=None):

    return evo.model(
        GS_instance(),
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
    )


def GN_sm(discrete_edges=None):

    return evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
    )
