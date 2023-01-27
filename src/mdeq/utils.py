import dataclasses
import json
import pathlib
import pickle
import re
import warnings

from dataclasses import asdict
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import numpy

from blosc2 import decompress
from cogent3 import get_app, make_table, open_data_store
from cogent3.app.composable import NotCompleted, define_app
from cogent3.app.sqlite_data_store import (
    _LOG_TABLE,
    OVERWRITE,
    DataStoreSqlite,
)
from cogent3.app.typing import AlignedSeqsType, SerialisableType
from cogent3.util import deserialise
from cogent3.util.dict_array import DictArray
from cogent3.util.misc import get_object_provenance
from scipy.interpolate import UnivariateSpline


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


def matches_type(dstore, types):
    types = (types,) if isinstance(types, str) else types
    rt = dstore.record_type
    if not rt:
        return True
    return any(rt.endswith(t) for t in types)


def configure_parallel(parallel: bool, mpi: int) -> dict:
    """returns parallel configuration settings for use as composable.apply_to(**config)"""
    mpi = None if mpi < 2 else mpi  # no point in MPI if < 2 processors
    parallel = True if mpi else parallel
    par_kw = dict(max_workers=mpi, use_mpi=True) if mpi else None

    return {"parallel": parallel, "par_kw": par_kw}


@define_app
def set_fg_edge(
    aln: AlignedSeqsType, fg_edge=None
) -> Union[SerialisableType, AlignedSeqsType]:
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
class CompressedValueOld:
    """container class to support delayed decompression of serialised data"""

    data: bytes

    @property
    def decompressed(self) -> str:
        if not self.data:
            return b""
        try:
            return decompress(self.data)
        except RuntimeError:
            return self.data
        except TypeError:
            return self.data.encode("utf8")

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


@dataclasses.dataclass
class CompressedValue:
    """container class to support delayed decompression of serialised data"""

    data: bytes
    unpickler = get_app("unpickle_it")
    decompress = get_app("decompress")

    @property
    def decompressed(self) -> bytes:
        if not self.data:
            return b""
        r = self.decompress(self.data)
        return r

    @property
    def as_primitive(self):
        """decompresses and then returns as primitive python types"""
        if not self.data:
            return ""
        r = self.unpickler(self.decompressed)
        return r

    @property
    def deserialised(self):
        r = deserialise.deserialise_object(self.as_primitive)
        return r


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
    return [p for p in indir.glob(pattern) if p.suffix == ".sqlitedb"]


def omit_suffixes_from_path(path: Path) -> str:
    """removes all components of stem after '.'"""
    return path.stem.split(".", maxsplit=1)[0]


def estimate_freq_null(
    pvalues: numpy.ndarray,
    use_log: bool = False,
    start: float = 0.05,
    stop: float = 0.96,
    step: float = 0.05,
    use_mse: bool = True,
) -> float:
    """estimate proportion of for which null hypothesis is true

    Parameters
    ----------
    pvalues
        series of p-values
    use_log
        fit spline using natural log transform
    start, stop, step
        used to produce the lambda series
    use_mse
        identifies the best lambda using the mean square error

    Returns
    -------
    Estimate of the proportion of p-values for which null is True

    Notes
    -----
    Based on description in

    JD Storey & R Tibshirani. Statistical significance for genomewide studies.
    Proc National Acad Sci 100, 9440–9445 (2003).

    and compared with results from the R q-value package at
    https://github.com/StoreyLab/qvalue

    MSE approach from '6. Automatically choosing λ'
    from Storey, Taylor, and Siegmund, 2004 and the R q-value package
    """
    pvalues = numpy.array(sorted(pvalues))
    if min(start, stop, step) <= 0 or max(start, stop) >= 1 or start > stop:
        raise ValueError(f"start, stop, step must all be positive with start < stop")

    if pvalues.max() <= stop:
        stop = 0.95 * pvalues.max()

    lambdas = numpy.arange(start, stop, step)
    intervals = numpy.digitize(pvalues, lambdas)
    cumsums = numpy.cumsum(numpy.bincount(intervals)[1:][::-1])
    denom = pvalues.shape[0] * (1 - lambdas[::-1])

    freq_null = cumsums / denom
    freq_null = freq_null[::-1]

    if use_mse:
        result = _minimise_mse(pvalues, lambdas, freq_null)
    else:
        result = _spline_fit(lambdas, freq_null, use_log)

    result = min(result, 1.0)
    if result < 0.0:
        warnings.warn("estimate of freq_null <= 0, setting to 1.0")
        result = 1.0

    return result


def _spline_fit(lambdas, freq_null, use_log):
    if use_log:
        freq_null = numpy.log(freq_null)
    spline = UnivariateSpline(lambdas, freq_null, k=3)
    result = spline(lambdas)[-1]
    if use_log:
        result = numpy.exp(result)
    return result


def _minimise_mse(pvalues, lambdas, freq_null):
    # returns the frequency that minimises the mean square error
    num = len(pvalues)
    fdr_val = numpy.quantile(freq_null, q=0.1)
    W = numpy.array([(pvalues > l).sum() for l in lambdas])
    a = W / (num ** 2 * (1 - lambdas) ** 2) * (1 - W / num) + (freq_null - fdr_val) ** 2
    return freq_null[a == a.min()][0]


def _get_composed_func_str_from_log(text: str) -> str:
    term = "composable function :"
    if term not in text:
        return "unnamed"
    line_prefix = text.split(maxsplit=1)[0]
    text = text.split(term, maxsplit=1)[1].split(line_prefix, maxsplit=1)[0]
    return " ".join(text.split())


def _reserialised(data: dict) -> dict:
    if hasattr(data, "to_rich_dict"):
        return data.to_rich_dict()
    elif isinstance(data, str):
        return data

    serialiser = get_app("to_primitive") + get_app("pickle_it") + get_app("compress")

    for k, v in data.items():
        if isinstance(v, dict):
            v = _reserialised(v)
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if len(e) == 2:
                    v[i][1] = _reserialised(v[i][1])
                else:
                    v[i] = _reserialised(v[i])
        elif isinstance(v, bytes):
            v = CompressedValueOld(v).deserialised
            v = serialiser(v)

        data[k] = v
    return data


def convert_db_to_new_sqlitedb(
    source: Path, dest: Optional[Path] = None, overwrite: bool = False
):
    """convert mdeq custom sqlitedb to cogent3 sqlitedb"""
    import os
    import sys

    from cogent3.app.composable import _make_logfile_name
    from cogent3.app.io_new import compress, pickle_it, to_primitive, write_db
    from scitrack import CachingLogger

    from mdeq.sqlite_data_store import ReadonlySqliteDataStore, sql_loader

    dest = dest or Path(source.parent) / f"{source.stem}-new.sqlitedb"

    if dest.exists() and not overwrite:
        raise IOError(f"cannot overwrite existing {str(dest)}")

    dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    new_dstore = DataStoreSqlite(source=dest, mode=OVERWRITE)
    serialiser = to_primitive() + pickle_it() + compress()
    new_writer = write_db(data_store=new_dstore, serialiser=serialiser)

    old_dstore = ReadonlySqliteDataStore(source=source)

    # this is the only way I've managed to suppress bloody deprecation warnings!
    with open(os.devnull, mode="w") as out:
        sys.stderr = out

        # we don't try and fully deserialise objects, just get back to
        # the python primitives
        old_loader = sql_loader(deserialiser=lambda x: x)
        for m in old_dstore.members:
            obj = old_loader(m)
            if new_dstore.record_type is None:
                # set the type dict entry directly, since we're not
                # deserialising into the original object
                new_dstore.db.execute(
                    "UPDATE state SET record_type=? WHERE state_id=1", (obj["type"],)
                )
            obj = _reserialised(obj)
            if not obj:
                raise AssertionError(f"{obj} not completed for {m}")

            _ = new_writer.main(data=obj, identifier=m.name)

        # but we do fully deserialise NotCompleted objects so they get
        # written properly
        old_loader = sql_loader()
        for m in old_dstore.incomplete:
            obj = old_loader(m)
            new_writer.main(identifier=m.name, data=obj)

    sys.stderr = sys.__stderr__

    # copy the logfile state
    assert len(old_dstore.logs) == 1
    log_record = old_dstore.db.execute("SELECT * from logs").fetchone()

    cmnd = f"UPDATE {_LOG_TABLE} SET date =?"
    values = (log_record["date"],)
    if log_data := old_dstore.logs[0].decompressed.decode("utf8"):
        # need to create a filename
        log_name = _get_composed_func_str_from_log(log_data)
        log_name = _make_logfile_name(log_name)
        cmnd = f"{cmnd}, log_name =?, data =?"
        values += (log_name, log_data)

    cmnd = f"{cmnd} WHERE log_id=?"
    values += (log_record["log_id"],)

    new_writer.data_store.db.execute(cmnd, values)

    assert len(old_dstore.incomplete) == len(
        new_dstore.not_completed
    ), f"FAILED to convert {str(source)}"
    assert len(old_dstore) == len(new_dstore.completed)
    if log_data:
        assert len(old_dstore.logs) == len(new_dstore.logs)

    lock_id = old_dstore._lock_pid
    new_dstore.close()

    LOGGER = CachingLogger(create_dir=True)
    log_file_path = source.parent / _make_logfile_name("convert_db_to_new_sqlitedb")
    LOGGER.log_file_path = log_file_path
    LOGGER.shutdown()

    new_dstore = DataStoreSqlite(source=dest, mode="a")
    new_dstore.write_log(unique_id=log_file_path.name, data=log_file_path.read_text())
    log_file_path.unlink()

    # copy lock status
    if lock_id:
        cmnd = f"UPDATE state SET lock_pid =? WHERE state_id == 1"
        new_dstore.db.execute(cmnd, (lock_id,))
    else:
        new_dstore.unlock()

    return new_dstore


@define_app
def upgrade_db(
    path: Union[Path, PosixPath, WindowsPath],
    rootdir,
    outdir,
    fn_sig,
    overwrite: bool = False,
) -> DataStoreSqlite:
    dest = outdir / path.parent.relative_to(rootdir) / f"{path.stem}{fn_sig}.sqlitedb"
    return convert_db_to_new_sqlitedb(source=path, dest=dest, overwrite=overwrite)


deserialiser = (
    get_app("decompress") + get_app("unpickle_it") + get_app("from_primitive")
)


def load_from_sqldb():
    return get_app("load_db", deserialiser=deserialiser)


serialiser = get_app("to_primitive") + get_app("pickle_it") + get_app("compress")


def write_to_sqldb(data_store, id_from_source=None):
    from cogent3.app.io_new import get_unique_id

    id_from_source = id_from_source or get_unique_id

    return get_app(
        "write_db",
        data_store=data_store,
        serialiser=serialiser,
        id_from_source=id_from_source,
    )


def summary_not_completed(dstore):
    from cogent3.app.data_store_new import summary_not_completeds

    return summary_not_completeds(dstore.not_completed, deserialise=deserialiser)


def db_status(inpath):
    _cmnd = re.compile(r"command_string\s+:")
    _params = re.compile(r"params\s+:")
    _path = re.compile(r"[A-Z]*[a-z]+Path\('[^']*'\)")
    _types = re.compile(r"\b(None|True|False)\b")
    dstore = open_data_store(inpath)
    record_type = dstore.record_type
    cmnds = []
    args = []
    dates = []
    params = []
    for log in dstore.logs:
        log = log.read().splitlines()
        if not log:
            continue
        timestamp = " ".join(log[0].split()[:2])
        dates.append(timestamp)
        for line in log:
            if _cmnd.search(line):
                line = _cmnd.split(line)[-1].split("\t")[0].strip().split()
                cmnd = pathlib.Path(line[0])
                cmnds.append(cmnd.name)
                args.append(" ".join(line[1:]))
                continue

            if _params.search(line):
                params.append(_params.split(line)[-1].strip())
                continue
    # clean up the params so that the value excludes
    for i, p in enumerate(params):
        for match in _path.findall(p):
            repl = match.split("'")[1]
            p = p.replace(match, f"{repl!r}")
        params[i] = p
    _type_map = {"None": "null", "True": "true", "False": "false"}
    for i, p in enumerate(params):
        p = re.sub("[<]module[^>]+[>]", "'module'", p)
        for match in _types.findall(p):
            pattern = f"\\b{match}\\b"
            p = re.sub(pattern, _type_map[match], p)
        p = p.replace("'", '"')
        try:
            params[i] = json.loads(p)
        except json.JSONDecodeError:
            params[i] = {"...": "json error decoding parameters"}
        else:
            params[i] = {k: v for k, v in params[i].items() if v != "module"}
    columns = ["date", "command", "args"]
    cmnds = make_table(
        header=columns,
        data=list(zip(dates, cmnds, args)),
        title=f"{str(inpath)!r} generated by",
    )

    rich_display(cmnds)

    rows = []
    for i, p in enumerate(params):
        rows.extend([[dates[i]] + list(item) for item in p.items()])

    columns = ["date", "param_name", "value"]
    all_params = make_table(
        header=columns,
        data=rows,
        title="Set parameters and default values",
    )

    rich_display(all_params)
    rich_display(
        make_table(header=["data type"], data=[[f"{record_type!r}"]], title="Contents")
    )

    t = dstore.describe
    t.title = "Content summary"
    rich_display(t)

    if len(dstore.not_completed) > 0:
        t = summary_not_completed(dstore)
        t.title = "Summary of incomplete records"
        rich_display(t)

    if len(dstore.completed) == 0:
        one = deserialiser(dstore.not_completed[0].read())
        print(
            "",
            "DataStore has only not completed members, displaying one.",
            one,
            sep="\n",
        )
