import re
import time

from cogent3 import get_model, make_tree
from cogent3.app import evo, io
from cogent3.app.composable import RAISE
from numpy.random import default_rng
from scitrack import CachingLogger

from kath_library.jsd import get_jsd
from kath_library.stationary_pi import get_stat_pi_via_eigen

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__maintainer__ = "Katherine Caley"


LOGGER = CachingLogger()


def scale_branch_lengths(lf, scale):

    len_rules = [a for a in lf.get_param_rules() if a["par_name"] == "length"]
    rule = len_rules[0]

    length = rule["init"]

    if scale * length > 0.6:
        scale = 0.6 / length

    lf.set_param_rule("length", value=scale * length)

    return lf


def generate_stationary_lf(aln, fg_edge=None, scale_branch=1):
    """
    aln : alignment of 3 sequences.
    fg_edge : String
        Name of the foregound edge which will be used to define the stationary dist.
        If None, it will be the edge from the ingroup with the max average JSD.

    Returns
    -------
    A likelihood function (lf) where the motif probs are set to the stationary distribution
    of the foreground (fg) edge.

    """
    if fg_edge is None:
        fg_edge, _, _ = get_jsd(aln)
    else:
        assert fg_edge in aln.names

    bg_edges = list({fg_edge} ^ set(aln.names))

    gn = evo.model(
        "GN",
        sm_args=dict(optimise_motif_probs=True),
        lf_args=dict(discrete_edges=bg_edges, expm="pade"),
    )

    result = gn(aln)
    lf = result.lf

    if not (scale_branch == 1):
        lf = scale_branch_lengths(lf, scale_branch)

    psub_fg = lf.get_psub_for_edge(fg_edge)
    stat_pi_fg = get_stat_pi_via_eigen(psub_fg)

    stationary_pi_rule = {
        base: stat_pi_fg[index] for index, base in enumerate(list(aln.moltype))
    }

    lf.set_motif_probs(stationary_pi_rule)

    return lf, fg_edge


def generate_equivelent_edge_lf(model_result, fg_edges):
    """
    model_result:
        gn model fit
    fg_edges
        list of string names of which edges to be equivalent
    """

    tree = model_result.alignment.quick_tree()
    gn = get_model("GN")
    lf = gn.make_likelihood_function(tree=tree)

    pattern = re.compile(r"[ACGT]>[ACGT]")
    param_rules = model_result.lf.get_param_rules()
    rate_rules = [
        rule
        for rule in param_rules
        if pattern.search(rule["par_name"]) and rule["edge"] == fg_edges[0]
    ]

    for rule in rate_rules:
        if rule["par_name"] == "T>G":
            break
        else:
            lf.set_param_rule(**rule)
            rule["edge"] = fg_edges[1]
            lf.set_param_rule(**rule)

    lf.set_motif_probs(model_result.lf.get_motif_probs())

    return lf


def generate_equivelent_edge_aln(model_result, length, num):
    seed = int(time.time())
    rng = default_rng(seed=seed)
    aln_id = model_result.source[42:-6]

    outpath = f"~/repos/data/microbial/equivalent/{aln_id}/{length}bp.tinydb"

    LOGGER.log_args()
    LOGGER.log_versions(["numpy", "cogent3", "kath_library"])
    LOGGER.log_file_path = f"~/repos/data/microbial/equivalent/{aln_id}/{length}bp.log"

    tip_dists = model_result.alignment.distance_matrix().to_dict()
    ingroup = min(tip_dists, key=lambda k: tip_dists[k])

    lf = generate_equivelent_edge_lf(model_result, ingroup)

    writer = io.write_db(
        outpath,
        create=True,
        if_exists=io.RAISE,
    )

    for n in range(1, num + 1):
        seq_seed = rng.choice(seed)
        syn_aln = lf.simulate_alignment(sequence_length=length, seed=seq_seed)

        syn_aln.info["seed"] = model_result.source
        syn_aln.info["ingroup"] = ingroup
        syn_aln.info["source"] = str(n) + ".json"

        writer.write(syn_aln, f"{n}.json")
    writer.data_store.close()


def generate_syn_aln(inpath, aln_id, outpath=None, num=10, length=None, scale_branch=1):
    seed = int(time.time())
    rng = default_rng(seed=seed)

    if outpath is None:
        outpath = f"~/repos/data/microbial/synthetic/{aln_id}/{length}bp/alns.tinydb"

    LOGGER.log_args()
    LOGGER.log_versions("numpy")
    LOGGER.log_versions("cogent3")
    LOGGER.input_file(inpath)
    LOGGER.log_file_path = (
        f"~/repos/data/microbial/synthetic/{aln_id}/{length}bp/alns.log"
    )

    dstore = io.get_data_store(inpath)
    selected = dstore.filtered(f"{aln_id}*")

    aln = io.load_db()(selected[0])
    lf, fg_edge = generate_stationary_lf(aln, scale_branch=scale_branch)

    if length is None:
        length = len(aln)

    writer = io.write_db(
        outpath,
        create=True,
        if_exists=RAISE,
    )

    for num in range(1, num + 1):
        seq_seed = rng.choice(seed)
        syn_aln = lf.simulate_alignment(sequence_length=length, seed=seq_seed)

        syn_aln.info["seed"] = aln_id
        syn_aln.info["source"] = str(num) + ".json"
        syn_aln.info["fg_edge"] = fg_edge

        writer.write(syn_aln, f"{num}.json")

    writer.data_store.close()
