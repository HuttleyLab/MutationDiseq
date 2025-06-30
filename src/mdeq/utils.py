import contextlib
import dataclasses
import json
import pathlib
import re
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy
from cogent3 import get_app, make_table, open_data_store
from cogent3.app import io as io_app
from cogent3.app.composable import NotCompleted, define_app, get_unique_id
from cogent3.app.typing import AlignedSeqsType, SerialisableType
from cogent3.util import deserialise
from cogent3.util.dict_array import DictArray
from cogent3.util.misc import get_object_provenance
from scipy.interpolate import UnivariateSpline

try:
    from wakepy.keep import running as keep_running

    # trap flaky behaviour on linux
    with keep_running():
        ...

except (NotImplementedError, ImportError):
    keep_running = contextlib.nullcontext


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
        raise NotImplementedError

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
    aln: AlignedSeqsType,
    fg_edge=None,
) -> SerialisableType | AlignedSeqsType:
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
            source=get_unique_id(aln),
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

    for row in zip(*columns, strict=False):
        rich_table.add_row(*row)

    console = Console()
    console.print(rich_table)


@dataclasses.dataclass
class CompressedValue:
    """container class to support delayed decompression of serialised data"""

    data: bytes
    unpickler = io_app.unpickle_it()
    decompress = io_app.decompress()

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
    indir: Path,
    pattern: str,
    recursive: bool,
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
        raise ValueError("start, stop, step must all be positive with start < stop")

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
    W = numpy.array([(pvalues > l).sum() for l in lambdas])  # noqa: E741
    a = W / (num**2 * (1 - lambdas) ** 2) * (1 - W / num) + (freq_null - fdr_val) ** 2
    return freq_null[a == a.min()][0]


deserialiser = io_app.decompress() + io_app.unpickle_it() + io_app.from_primitive()


def load_from_sqldb():
    return get_app("load_db", deserialiser=deserialiser)


serialiser = io_app.to_primitive() + io_app.pickle_it() + io_app.compress()


def write_to_sqldb(data_store, id_from_source=None):
    from cogent3.app.io import get_unique_id

    id_from_source = id_from_source or get_unique_id

    return get_app(
        "write_db",
        data_store=data_store,
        serialiser=serialiser,
        id_from_source=id_from_source,
    )


def summary_not_completed(dstore):
    from cogent3.app.data_store import summary_not_completeds

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
        data=list(zip(dates, cmnds, args, strict=False)),
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
        make_table(header=["data type"], data=[[f"{record_type!r}"]], title="Contents"),
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
