import json

from dataclasses import asdict

from cogent3.util.dict_array import DictArray
from cogent3.util.misc import get_object_provenance


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]


def get_foreground(aln):
    """returns fg_edge value from info attribute."""
    try:
        fg = aln.info.get("fg_edge", None)
    except AttributeError:
        fg = None
    return fg


def foreground_from_jsd(aln):
    """returns the ingroup lineage with maximal JSD.

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
        result = {
            "type": get_object_provenance(self),
            "source": self.source,
        }
        return {**result, **asdict(self)}

    def to_json(self):
        return json.dumps(self.to_rich_dict())

    @classmethod
    def from_json(cls, data):
        """constructor from json data."""
        data.pop("type", None)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict):
        """constructor from dict data."""
        data.pop("type", None)
        return cls(**data)


def get_obj_type(dstore):
    """returns the record type in dstore"""
    from cogent3.app import data_store

    data = json.loads(dstore[0].read())
    return data["type"].split(".")[-1]


def configure_parallel(parallel: bool, mpi: int) -> dict:
    """returns parallel configuration settings for use as composable.apply_to(**config)"""
    mpi = None if mpi < 2 else mpi  # no point in MPI if < 2 processors
    parallel = True if mpi else parallel
    par_kw = dict(max_workers=mpi, use_mpi=True) if mpi else None

    return {"parallel": parallel, "par_kw": par_kw}
