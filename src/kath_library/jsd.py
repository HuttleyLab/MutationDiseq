from accupy import fsum as sum
from cogent3.app.composable import SERIALISABLE_TYPE, user_function
from cogent3.app.result import generic_result
from cogent3.core.profile import safe_p_log_p
from cogent3.util.dict_array import DictArray
from numpy import array as np_array

from kath_library.stationary_pi import get_stat_pi_via_brute, get_stat_pi_via_eigen

__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley"]
__version__ = "2021.07.01"
__maintainer__ = "Katherine Caley"
__email__ = "katherine.caley@anu.edu.au"
__status__ = "develop"


# todo change the terminology from ingroup in the cases where its not the ingroup, maybe change to pair?
def get_jsd(aln, edge=None):
    """

    Parameters
    ----------
    aln : 3 sequence alignment
    edge :  if None, defaults to the jsd of the ingroup, returned edge decided from highest jsd with the outgroup.
            if "max", returns the maximum pairwise jsd, returned edge has highest average pairwise jsd.
            if given an edge (as a string), will return the maximum pairwise jsd of that edge.

    Returns
    -------
            edge: the foreground edge, unless edge="max" this is not guaranteed.
            ingroup: tuple containing the names of edges from the jsd calculation.
            jsd: the jsd value for the ingroup.
    """

    freqs = aln.counts_per_seq().to_freq_array()
    pwise_jsd = freqs.pairwise_jsd()
    darr = DictArray(pwise_jsd)
    total_jsd = darr.row_sum().to_dict()

    if edge is None:
        tip_dists = aln.distance_matrix().to_dict()
        ingroup = min(tip_dists, key=lambda k: tip_dists[k])
        jsd = pwise_jsd[ingroup]
        total_jsd = {key: total_jsd[key] for key in ingroup}
        edge = max(total_jsd, key=lambda k: total_jsd[k])

    elif edge == "max":
        ingroup = max(pwise_jsd, key=lambda k: pwise_jsd[k])
        jsd = pwise_jsd[ingroup]
        total_jsd = {key: total_jsd[key] for key in ingroup}
        edge = max(total_jsd, key=lambda k: total_jsd[k])

    else:
        assert edge in aln.names, "fg_edge input name is not included in alignment"
        keys = [tup for tup in pwise_jsd.keys() if edge in tup]
        pwise_jsd = {key: pwise_jsd[key] for key in keys}
        ingroup = max(pwise_jsd, key=lambda k: pwise_jsd[k])
        jsd = pwise_jsd[ingroup]

    return edge, ingroup, jsd


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
    assert (
        edge in model_result.alignment.names
    ), "edge input name is not included in model_result"

    lf = model_result.lf

    if stat_pi:
        psub_fg = lf.get_psub_for_edge(edge)
        try:
            stat_pi_fg = get_stat_pi_via_eigen(psub_fg)
        except ArithmeticError:
            stat_pi_fg = get_stat_pi_via_brute(psub_fg, lf.get_motif_probs())
        entropy = sum(safe_p_log_p(stat_pi_fg))
    else:
        entropy = sum(safe_p_log_p(np_array(list(lf.get_motif_probs()))))

    return entropy


def _jsd_fg_edge(aln, fg_edge=None):
    edge, ingroup, jsd = get_jsd(aln, edge=fg_edge)
    result = generic_result(source=aln.info.source)
    result.update([("jsd", jsd), ("foreground", edge)])
    return result


jsd_fg_edge = user_function(
    _jsd_fg_edge, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)


def _max_jsd_tree(aln):
    edge, ingroup, jsd = get_jsd(aln, edge="max")
    result = generic_result(source=aln.info.source)
    result.update([("edge", edge), ("edges", ingroup), ("jsd", jsd)])
    return result


max_jsd_tree = user_function(
    _max_jsd_tree, input_types=SERIALISABLE_TYPE, output_types=SERIALISABLE_TYPE
)
