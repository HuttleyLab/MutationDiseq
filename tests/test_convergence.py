from cogent3.app import evo, io
from kath_library.convergence import convergence


def test_convergence_construction():

    dstore = io.get_data_store(
        "~/repos/data/microbial/synthetic/758_443154_73021/3000bp.tinydb"
    )
    loader = io.load_db()
    aln1 = loader(dstore[0])
    fg_edge = aln1.info.fg_edge

    if fg_edge is None:
        raise TypeError("Alignment needs a info.fg_edge attribute")

    bg_edges = list({fg_edge} ^ set(aln1.names))

    GN = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        opt_args=dict(max_restarts=5, tolerance=1e-8),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )

    result = GN(aln1)

    Q = result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)

    convergence(Q)
