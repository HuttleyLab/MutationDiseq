from cogent3.util.dict_array import DictArray
from numpy import array


def get_pi_0(model_result):
    return model_result.lf.get_motif_probs().to_array()


def get_pi_tip(model_result, fg_edge):
    Q_darray = model_result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)
    pi_darray = model_result.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi = array([pi_darray[i] for i in Q_darray.keys()])

    return pi


def get_foreground(aln):
    try:
        fg_edge = aln.info.fg_edge
    except AttributeError:
        fg_edge = None

    if fg_edge is not None:
        return fg_edge

    freqs = aln.counts_per_seq().to_freq_array()
    jsd_pwise = freqs.pairwise_jsd()
    darr = DictArray(jsd_pwise)
    jsd_totals = darr.row_sum().to_dict()

    tip_dists = aln.distance_matrix().to_dict()
    ingroup = min(tip_dists, key=lambda k: tip_dists[k])
    jsd_totals = {key: jsd_totals[key] for key in ingroup}
    fg_edge = max(jsd_totals, key=lambda k: jsd_totals[k])

    return fg_edge
