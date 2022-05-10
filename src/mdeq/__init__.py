"""mdeq: mutation disequilibrium analysis tools."""

# following line to stop automatic threading by numpy
from mdeq import _block_threading  # isort: skip
import inspect
import sys

from collections import defaultdict
from pathlib import Path
from warnings import filterwarnings

import click

from cogent3 import make_table
from cogent3.app import io
from rich.console import Console
from rich.progress import Progress, track
from scitrack import CachingLogger

import mdeq._click_options as _cli_opt

# required to ensure registration of define substitution models
from mdeq import model as _model
from mdeq import sqlite_data_store as _sqldstore
from mdeq.adjacent import load_data_group, physically_adjacent
from mdeq.bootstrap import bootstrap_toe
from mdeq.control import control_generator, select_model_result
from mdeq.convergence import bootstrap_to_nabla
from mdeq.eop import (
    ALT_AEOP,
    ALT_TEOP,
    NULL_AEOP,
    NULL_TEOP,
    adjacent_eop,
    temporal_eop,
)
from mdeq.sqlite_data_store import sql_loader, sql_writer
from mdeq.toe import ALT_TOE, NULL_TOE
from mdeq.utils import (
    configure_parallel,
    get_obj_type,
    omit_suffixes_from_path,
    paths_to_sqlitedbs_matching,
    set_fg_edge,
)


try:
    from wakepy import set_keepawake, unset_keepawake
except NotImplementedError:
    # may not be installed, or on linux where this library doesn't work
    make_none = type(None)

    set_keepawake, unset_keepawake = make_none, make_none


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

__version__ = "2021.12.20"

filterwarnings("ignore", "Not using MPI")
filterwarnings("ignore", "Unexpected warning from scipy")
filterwarnings("ignore", "using slow exponentiator")
filterwarnings("ignore", ".*creased to keep within bounds")
filterwarnings("ignore", "Used mean of.*", module="cogent3")

_min_version = (3, 10)
if sys.version_info < _min_version:
    PY_VERSION = ".".join([str(n) for n in sys.version_info])
    _min_version = ".".join(_min_version)
    raise RuntimeError(
        f"Python-{_min_version} or greater is required, Python-{PY_VERSION} used."
    )


def get_opt_settings(testrun):
    """create optimisation settings."""
    return (
        {"max_restarts": 1, "limit_action": "ignore", "max_evaluations": 10}
        if testrun
        else None
    )


@click.group()
@click.version_option(__version__)
def main():
    """mdeq: mutation disequilibrium analysis tools."""
    pass


@main.command()
@_cli_opt._inpath
@_cli_opt._gene_order
@_cli_opt._outpath
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
@_cli_opt._testrun
def make_adjacent(inpath, gene_order, outpath, limit, overwrite, verbose, testrun):
    """makes sqlitedb of adjacent alignment records."""
    LOGGER = CachingLogger(create_dir=True)

    LOGGER.log_file_path = outpath.parent / f"{outpath.stem}-mdeq-make_adjacent.log"
    LOGGER.log_args()

    # we get member names from input dstore
    dstore = io.get_data_store(inpath, limit=limit)
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )

    sample_ids = {m.name.replace(".json", "") for m in dstore}
    paired = physically_adjacent(gene_order, sample_ids)
    # make the grouped data app
    group_loader = load_data_group(inpath)
    for pair in track(paired, refresh_per_second=5):
        record = group_loader(pair)
        writer(record)

    log_file_path = LOGGER.log_file_path
    LOGGER.shutdown()
    writer.data_store.add_file(log_file_path, cleanup=True, keep_suffix=True)
    writer.data_store.close()

    writer.data_store.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command()
@_cli_opt._inpath
@_cli_opt._treepath
@_cli_opt._outpath
@_cli_opt._just_continuous
@_cli_opt._fg_edge
@_cli_opt._sequential
@_cli_opt._num_reps
@_cli_opt._parallel
@_cli_opt._mpi
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
@_cli_opt._testrun
def toe(
    inpath,
    treepath,
    outpath,
    just_continuous,
    fg_edge,
    sequential,
    num_reps,
    parallel,
    mpi,
    limit,
    overwrite,
    verbose,
    testrun,
):
    """test of existence of mutation equilibrium."""
    # todo need a separate command to apply foreground_from_jsd() to an
    #  alignment for decorating alignments with the foreground edge
    # or check alignment.info for a fg_edge key -- all synthetic data
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-toe.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        exit(1)

    loader = sql_loader()

    # check consistency of just_continuous / fg_edge / aln.info
    _aln = loader(dstore[0])
    if just_continuous and fg_edge is not None:
        click.secho(
            f"WARN: setting just_continuous overrides {fg_edge!r} setting", fg="yellow"
        )

    # if fg_edge is specified then this value is checked for existence in alignment
    if fg_edge is not None:
        if fg_edge not in _aln.names:
            click.secho(f"FAIL: {fg_edge!r} name not present in {_aln.names}", fg="red")
            exit(1)

        info_val = _aln.info.get("fg_edge", None)
        if info_val and info_val != fg_edge:
            click.secho(
                f"WARN: fg_edge={fg_edge!r} will override aln.info.fg_edge={info_val!r}",
                fg="yellow",
            )

    inject_fg = set_fg_edge(fg_edge=fg_edge) if fg_edge else None
    opt_args = get_opt_settings(testrun)
    bstrapper = bootstrap_toe(
        tree=treepath,
        num_reps=num_reps,
        opt_args=opt_args,
        sequential=sequential,
        just_continuous=just_continuous,
    )
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    if inject_fg:
        app = loader + inject_fg + bstrapper + writer
    else:
        app = loader + bstrapper + writer

    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    if mpi:
        kwargs["par_kw"]["chunksize"] = 1
    app.apply_to(
        dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2, **kwargs
    )
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command()
@_cli_opt._inpath
@_cli_opt._outpath
@_cli_opt._treepath
@_cli_opt._edge_names
@_cli_opt._parallel
@_cli_opt._mpi
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
@_cli_opt._testrun
def teop(
    inpath,
    outpath,
    treepath,
    edge_names,
    parallel,
    mpi,
    limit,
    overwrite,
    verbose,
    testrun,
):
    """between branch equivalence of mutation process test"""
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-teop.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        exit(1)

    # construct hypothesis app, null constrains edge_names to same process
    loader = sql_loader()
    opt_args = get_opt_settings(testrun)
    teop = temporal_eop(edge_names, tree=treepath, opt_args=opt_args)
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + teop + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    process.apply_to(
        dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2, **kwargs
    )
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command()
@_cli_opt._inpath
@_cli_opt._outpath
@_cli_opt._treepath
@_cli_opt._share_mprobs
@_cli_opt._parallel
@_cli_opt._mpi
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
@_cli_opt._testrun
def aeop(
    inpath,
    outpath,
    treepath,
    share_mprobs,
    parallel,
    mpi,
    limit,
    overwrite,
    verbose,
    testrun,
):
    """between loci equivalence of mutation process test"""
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-aeop.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("grouped",)
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        exit(1)

    loader = sql_loader()
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    test_adjacent = adjacent_eop(
        tree=treepath, opt_args=get_opt_settings(testrun), share_mprobs=share_mprobs
    )
    process = loader + test_adjacent + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    _ = process.apply_to(
        dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 1, **kwargs
    )
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command()
@_cli_opt._inpath
@_cli_opt._outpath
@_cli_opt._wrt_nstat
@_cli_opt._parallel
@_cli_opt._mpi
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
def convergence(inpath, outpath, wrt_nstat, parallel, mpi, limit, overwrite, verbose):
    """estimates convergence towards mutation equilibrium."""
    set_keepawake(keep_screen_awake=False)

    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-convergence.log"
    LOGGER.log_args()
    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("compact_bootstrap_result",)
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        exit(1)

    loader = sql_loader(fully_deserialise=True)
    to_delta_nabla = bootstrap_to_nabla(wrt_nstat=wrt_nstat)
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + to_delta_nabla + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    r = process.apply_to(
        dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 1, **kwargs
    )
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")
    unset_keepawake()


@main.command()
@_cli_opt._inpath
@_cli_opt._outdir
@_cli_opt._analysis
@_cli_opt._controls
@_cli_opt._sample_size
@_cli_opt._seed
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
@_cli_opt._testrun
def make_controls(
    inpath,
    outdir,
    analysis,
    controls,
    sample_size,
    seed,
    limit,
    overwrite,
    verbose,
    testrun,
):
    """simulate negative and positive controls

    Notes

    A single simulated record is produced for an input record.
    """
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_args()

    ctl_txt = "neg_control" if controls == "-ve" else "pos_control"
    outpath = outdir / Path(f"{analysis}-{ctl_txt}-{inpath.stem}.sqlitedb")
    LOGGER.log_file_path = outdir / f"{outpath.stem}-make_controls.log"

    # create loader, read a single result and validate the type matches the controls choice
    # validate the model choice too
    dstore = io.get_data_store(inpath, limit=limit)
    result_types = {
        "teop": "hypothesis_result",
        "aeop": "hypothesis_result",
        "toe": "compact_bootstrap_result",
        "single-model": "model_result",
    }
    control_name = {
        "aeop": {"-ve": NULL_AEOP, "+ve": ALT_AEOP},
        "teop": {"-ve": NULL_TEOP, "+ve": ALT_TEOP},
        "toe": {"-ve": NULL_TOE, "+ve": ALT_TOE},
        "single-model": {"-ve": "", "+ve": ""},
    }
    record_type = get_obj_type(dstore)
    if record_type != result_types[analysis]:
        click.secho(
            f"object type in {inpath!r} does not match expected "
            f"{result_types[analysis]!r} for analysis {analysis!r}",
            fg="red",
        )
        exit(1)

    model_name = control_name[analysis][controls]
    model_selector = select_model_result(model_name)

    loader = sql_loader()
    generator = control_generator(model_selector, seed=seed)

    # now use that rng to randomly select sample_size from dstore
    if verbose:
        print(f"{dstore!r}")

    if sample_size is not None:
        # sample without replacement
        dstore = [
            dstore[i] for i in generator.rng.sample(range(len(dstore)), sample_size)
        ]
        if verbose > 3:
            print(f"{dstore!r}")

    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    proc = loader + generator + writer
    proc.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


# todo postprocess functions, generate figures, tabulate data
@main.command()
@_cli_opt._inpath
def db_summary(inpath):
    """displays summary information about a db"""
    import json
    import pathlib
    import re

    from cogent3 import make_table

    from mdeq.utils import rich_display

    _cmnd = re.compile(r"command_string\s+:")
    _params = re.compile(r"params\s+:")
    _path = re.compile(r"[A-Z]*[a-z]+Path\('[^']*'\)")
    _types = re.compile(r"\b(None|True|False)\b")

    dstore = io.get_data_store(inpath)
    record_type = get_obj_type(dstore)

    cmnds = []
    args = []
    params = []
    timestamp = ""
    for log in dstore.logs:
        log = log.deserialised.splitlines()
        if not log:
            continue
        timestamp = " ".join(log[0].split()[:2])
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

    columns = ["num", "command", "args"]
    cmnds = make_table(
        header=columns,
        data=list(zip(range(len(cmnds)), cmnds, args)),
        title=f"'{inpath}' was generated on {timestamp} by",
    )
    cmnds = cmnds if cmnds.shape[0] > 1 else cmnds[:, columns[1:]]
    rich_display(cmnds)

    rows = []
    for i, p in enumerate(params):
        rows.extend([[i] + list(item) for item in p.items()])

    columns = ["num", "param_name", "value"]
    all_params = make_table(
        header=columns,
        data=rows,
        title="Set parameters and default values",
    )
    all_params = all_params if cmnds.shape[0] > 1 else all_params[:, columns[1:]]
    rich_display(all_params)
    rich_display(
        make_table(header=["data type"], data=[[f"{record_type!r}"]], title="Contents")
    )
    t = dstore.describe
    t.title = "Content summary"
    rich_display(t)
    if len(dstore.incomplete) > 0:
        t = dstore.summary_incomplete
        t.title = "Summary of incomplete records"
        rich_display(t)


@main.command()
@_cli_opt._indir
@_cli_opt._pattern
@_cli_opt._recursive
@_cli_opt._outdir
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
def extract_pvalues(indir, pattern, recursive, outdir, limit, overwrite, verbose):
    """extracts p-values from TOE sqlitedb results

    generates a tsv with same name in same location"""
    console = Console()
    if verbose:
        console.print(f"[blue]{indir}")

    data_type = "compact_bootstrap_result"

    reader = sql_loader()
    paths = paths_to_sqlitedbs_matching(indir, pattern, recursive)
    if not paths:
        console.print(
            f"[red]EXIT: no paths found for {indir=}, {recursive=!r}, " f"{pattern=!r}"
        )
        exit(1)

    for path in paths:
        if outdir:
            outpath = outdir / f"{path.stem}.tsv"
        else:
            outpath = path.parent / f"{path.stem}.tsv"
        if outpath.exists() and not overwrite:
            if verbose:
                console.print(f"[green]{outpath} exists, skipping")
            continue

        dstore = io.get_data_store(path, limit=limit)
        record_type = get_obj_type(dstore)
        if record_type != data_type:
            console.print(
                "[yellow]SKIPPED: "
                f"record type {record_type!r} in '{path}' does not match "
                f"expected {data_type!r}",
            )
            continue

        data = defaultdict(list)
        for m in track(dstore):
            r = reader(m)
            if r.observed.pvalue is None:
                continue
            data["name"].append(m.name)
            data["chisq_pval"].append(r.observed.pvalue)
            data["bootstrap_pval"].append(r.pvalue)

        table = make_table(data=data)
        table.write(outpath)
    console.print("[green]Done!")


@main.command()
@_cli_opt._inpath
@_cli_opt._outpath
@_cli_opt._window
@_cli_opt._step
@_cli_opt._min_length
@_cli_opt._overwrite
@_cli_opt._verbose
def slide(
    inpath,
    outpath,
    window_size,
    step,
    min_length,
    overwrite,
    verbose,
):
    """generate window sized sub-alignments. The source of each sub-alignment
    is recorded as parent source with index position."""
    # or check alignment.info for a fg_edge key -- all synthetic data
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-slide.log"
    LOGGER.log_args()

    console = Console()
    if window_size < min_length:
        console.print(
            f"[red] {window_size=} is less than {min_length=}, should be other way around"
        )
        exit(1)

    dstore = io.get_data_store(inpath)
    expected_types = ("ArrayAlignment", "Alignment")
    if get_obj_type(dstore) not in expected_types:
        console.print(f"[red]records not one of the expected types {expected_types}")
        exit(1)

    loader = sql_loader()
    writer = sql_writer(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    with Progress() as progress:
        alignments = progress.add_task("[green]Alignment...", total=len(dstore))
        for i, member in enumerate(dstore):
            progress.update(alignments, completed=i + 1)
            n = omit_suffixes_from_path(Path(member.name))
            aln = loader(member)
            num_windows = len(aln) - window_size + 1
            windows = progress.add_task(
                "[blue]slide...", total=num_windows, transient=True
            )
            for start in range(0, num_windows, step):
                progress.update(windows, completed=start + 1)
                sliced = aln[start : start + window_size]
                sub = sliced.no_degenerates()
                if sub is None or len(sub) < min_length:
                    continue

                sub.info.source = f"{n}-{start}"
                sub.info.index = start
                # write it
                writer(sub)
            progress.remove_task(windows)

    log_file_path = LOGGER.log_file_path
    LOGGER.shutdown()
    writer.data_store.add_log(log_file_path, cleanup=True)
    console.print("[green]Done!")


if __name__ == "__main__":
    main()
