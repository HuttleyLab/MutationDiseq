from cogent3.app import evo, io

from kath_library.lrt import GS_mod


def create_bootstrap_app(num_reps=5):
    GS = evo.model(
        GS_mod(),
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=["Chimp, Gorilla"], expm="pade"),
    )

    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=["Chimp, Gorilla"], expm="pade"),
    )

    hyp = evo.hypothesis(GS, GN, sequential=False)
    bootie = evo.bootstrap(hyp, num_reps)

    return bootie
