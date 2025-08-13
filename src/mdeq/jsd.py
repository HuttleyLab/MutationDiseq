from cogent3.core.profile import safe_p_log_p
from cogent3.maths.measure import jsd as jsd_func
from cogent3.util.dict_array import DictArray

from mdeq.numeric import sum
from mdeq.stationary_pi import get_stat_pi_via_brute


def get_jsd(aln, edge=None, evaluate="ingroup"):
    """
     Parameters
    ----------
    aln : 3 sequence alignment
    edge : str for which taxa is the primary edge to be used in this analysis
            if None, defaults to foreground edge
    evaluate : determines which comparison to return the jsd value
                defaults to "ingroup", which returns the jsd of the ingroup,
                "total" returns the total jsd
                "all" return a dicts of all of the possible jsd evaluations
    """
    freqs = aln.counts_per_seq().to_freq_array()
    jsd_pwise = freqs.pairwise_jsd()
    darr = DictArray(jsd_pwise)
    jsd_totals = darr.row_sum().to_dict()

    if edge is None:
        tip_dists = aln.distance_matrix().to_dict()
        ingroup = min(tip_dists, key=lambda k: tip_dists[k])
        jsd_totals = {key: jsd_totals[key] for key in ingroup}
        edge = max(jsd_totals, key=lambda k: jsd_totals[k])
        if evaluate == "ingroup":
            jsd = jsd_pwise[ingroup]
            return edge, ingroup, jsd
    else:
        assert edge in aln.names, "fg_edge input name is not included in alignment"

    if evaluate == "ingroup":
        tip_dists = aln.distance_matrix().to_dict()
        ingroup = min(tip_dists, key=lambda k: tip_dists[k])

        assert edge in ingroup, (
            'evaluate="ingroup" is not valid if given edge is not in ingroup'
        )

        jsd = jsd_pwise[ingroup]
        return edge, ingroup, jsd

    if evaluate == "total":
        jsd = jsd_func(freqs[0], freqs[1], freqs[2])
        return edge, (aln.names[0], aln.names[1], aln.names[2]), jsd

    if evaluate == "max":
        keys = [tup for tup in jsd_pwise.keys() if edge in tup]
        jsd_pwise = {key: jsd_pwise[key] for key in keys}
        max_pair = max(jsd_pwise, key=lambda k: jsd_pwise[k])
        jsd = jsd_pwise[max_pair]
        return edge, max_pair, jsd

    if evaluate == "all":
        jsds = {}

        jsds["total_jsd"] = jsd_func(freqs[0], freqs[1], freqs[2])

        tip_dists = aln.distance_matrix().to_dict()
        ingroup = min(tip_dists, key=lambda k: tip_dists[k])
        jsds["ingroup_jsd"] = jsd_pwise[ingroup]

        keys = [tup for tup in jsd_pwise.keys() if edge in tup]
        jsd_pwise = {key: jsd_pwise[key] for key in keys}
        max_pair = max(jsd_pwise, key=lambda k: jsd_pwise[k])
        jsds["max_jsd"] = jsd_pwise[max_pair]

        return edge, ingroup, jsds


def get_entropy(model_result, edge, stat_pi=True):
    """

    Parameters
    ----------
    model_result : Storage of model results. Cogent3 type.
    edge : edge from which nucleotide distribution is used.
    stat_pi : if True, entropy is calculated from stationary pi distribution.
                else, entropy is calculated from observed pi distribution.

    Returns
    -------
            Entropy of a nucleotide distribution.
    """
    assert edge in model_result.alignment.names, (
        "edge input name is not included in model_result"
    )

    lf = model_result.lf
    pi = lf.get_motif_probs().to_array()

    if stat_pi:
        psub_fg = lf.get_psub_for_edge(edge).to_array()
        stat_pi_fg = get_stat_pi_via_brute(psub_fg, pi)
        entropy = sum(safe_p_log_p(stat_pi_fg))
    else:
        entropy = sum(safe_p_log_p(pi))

    return entropy
