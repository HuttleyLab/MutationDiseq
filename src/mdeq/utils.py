import json

from cogent3.util.dict_array import DictArray
from cogent3.util.misc import get_object_provenance
from numpy import array


def get_pi_0(model_result):
    return model_result.lf.get_motif_probs().to_array()


def get_pi_tip(model_result, fg_edge):
    Q_darray = model_result.lf.get_rate_matrix_for_edge(fg_edge, calibrated=False)
    pi_darray = model_result.alignment.counts_per_seq().to_freq_array()[fg_edge]

    pi = array([pi_darray[i] for i in Q_darray.keys()])

    return pi


def get_foreground(aln):
    """returns fg_edge value from info attribute"""
    try:
        fg = aln.info.get("fg_edge", None)
    except AttributeError:
        fg = None
    return fg


def foreground_from_jsd(aln):
    """returns the ingroup lineage with maximal JSD

    Notes
    -----
    Identifies the ingroup based on conventional genetic distance,
    identifies ingroup which has maximal JSD from the rest.
    """
    if aln.num_seqs != 3:
        raise NotImplementedError()

    freqs = aln.counts_per_seq().to_freq_array()
    jsd_pwise = freqs.pairwise_jsd()
    darr = DictArray(jsd_pwise)
    jsd_totals = darr.row_sum().to_dict()
    tip_dists = aln.distance_matrix().to_dict()
    ingroup = min(tip_dists, key=lambda k: tip_dists[k])
    jsd_totals = {key: jsd_totals[key] for key in ingroup}
    return max(jsd_totals, key=lambda k: jsd_totals[k])


class SerialisableMixin:
    def to_rich_dict(self):
        return {
            "type": get_object_provenance(self),
            "source": self.source,
        }

    def to_json(self):
        return json.dumps(self.to_rich_dict())
