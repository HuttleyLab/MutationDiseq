from cogent3 import get_moltype
from cogent3.app import evo
from cogent3.evolve.ns_substitution_model import GeneralStationary


def GS_instance():
    """
    A General Stationary Nucleotide substitution model instance.
    """
    GS = GeneralStationary(
        get_moltype("dna").alphabet, optimise_motif_probs=True, name="GS"
    )

    return GS


def GS_sm(discrete_edges=[]):

    GS = evo.model(
        GS_instance(),
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
    )

    return GS


def GN_sm(discrete_edges=[]):

    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=discrete_edges, expm="pade"),
    )

    return GN
