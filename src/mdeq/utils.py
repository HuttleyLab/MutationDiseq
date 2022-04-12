import dataclasses
import json
import pickle

from dataclasses import asdict
from pathlib import Path

from blosc2 import decompress
from cogent3.app.composable import (
    ALIGNED_TYPE,
    SERIALISABLE_TYPE,
    NotCompleted,
    appify,
)
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
    if len(dstore) == 0:
        return None
    data = dstore[0].read()
    if isinstance(data, str):
        data = json.loads(data)
        return data["type"].split(".")[-1]

    # already a dict with compressed values
    return data["type"].deserialised.split(".")[-1]


def configure_parallel(parallel: bool, mpi: int) -> dict:
    """returns parallel configuration settings for use as composable.apply_to(**config)"""
    mpi = None if mpi < 2 else mpi  # no point in MPI if < 2 processors
    parallel = True if mpi else parallel
    par_kw = dict(max_workers=mpi, use_mpi=True) if mpi else None

    return {"parallel": parallel, "par_kw": par_kw}


@appify((ALIGNED_TYPE, SERIALISABLE_TYPE), (ALIGNED_TYPE, SERIALISABLE_TYPE))
def set_fg_edge(aln, fg_edge=None):
    """sets aln.info_fg_edge to fg_edge"""
    if fg_edge is None:
        raise ValueError("fg_edge not set")

    if isinstance(aln, NotCompleted):
        return aln

    assert fg_edge is not None
    if fg_edge not in aln.names:
        return NotCompleted(
            "ERROR",
            set_fg_edge.__name__,
            f"{fg_edge!r} not in {aln.names}",
            source=aln.info.source,
        )

    aln.info.fg_edge = fg_edge
    return aln


def rich_display(c3t, title_justify="left"):
    """converts a cogent3 Table to a Rich Table and displays it"""
    from cogent3.format.table import formatted_array
    from rich.console import Console
    from rich.table import Table

    cols = c3t.columns
    columns = [formatted_array(cols[c], pad=False)[0] for c in c3t.header]
    rich_table = Table(
        title=c3t.title,
        highlight=True,
        title_justify=title_justify,
        title_style="bold blue",
    )
    for col in c3t.header:
        numeric_type = any(v in cols[col].dtype.name for v in ("int", "float"))
        j = "right" if numeric_type else "left"
        rich_table.add_column(col, justify=j, no_wrap=numeric_type)

    for row in zip(*columns):
        rich_table.add_row(*row)

    console = Console()
    console.print(rich_table)


@dataclasses.dataclass
class CompressedValue:
    """container class to support delayed decompression of serialised data"""

    data: bytes

    @property
    def decompressed(self) -> str:
        return decompress(self.data) if self.data else ""

    @property
    def deserialised(self):
        """decompresses and then deserialises"""
        if not self.data:
            return ""
        try:
            return json.loads(self.decompressed.decode("utf8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return pickle.loads(self.decompressed)
            except pickle.UnpicklingError:
                # a compressed plain string
                return self.decompressed.decode("utf8")


def paths_to_sqlitedbs_matching(
    indir: Path, pattern: str, recursive: bool
) -> list[Path]:
    """finds paths matching pattern in indir

    Parameters
    ----------
    indir : Path
        root directory to search within
    pattern : str
        glob pattern, inserted as {pattern}.sqlitedb. Defaults to defaults to *.sqlitedb

    recursive : bool
        descend into sub-directories
    """
    if not pattern:
        pattern = "**/*.sqlitedb" if recursive else "*.sqlitedb"
    else:
        pattern = f"**/{pattern}.sqlitedb" if recursive else pattern
    return list(indir.glob(pattern))
