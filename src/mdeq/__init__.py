"""mdeq: mutation disequilibrium analysis tools."""

# following line to stop automatic threading by numpy
from mdeq import _block_threading  # noqa: F401 isort: skip
import inspect
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import reduce
from operator import add
from pathlib import Path
from warnings import filterwarnings

import click
import trogon
from cogent3 import get_app, make_table, open_data_store
from rich.console import Console
from rich.progress import Progress, track
from scitrack import CachingLogger

import mdeq._click_options as _cli_opt

# required to ensure registration of define substitution models
from mdeq import model as _model  # noqa: F401
from mdeq.adjacent import load_data_group, physically_adjacent
from mdeq.bootstrap import bootstrap_toe
from mdeq.control import control_generator, select_model_result
from mdeq.convergence import bootstrap_to_nabla, nabla_c_table
from mdeq.eop import (
    ALT_AEOP,
    ALT_TEOP,
    NULL_AEOP,
    NULL_TEOP,
    adjacent_eop,
    temporal_eop,
)
from mdeq.toe import ALT_TOE, NULL_TOE
from mdeq.utils import (
    configure_parallel,
    db_status,
    keep_running,
    load_from_sqldb,
    matches_type,
    omit_suffixes_from_path,
    paths_to_sqlitedbs_matching,
    set_fg_edge,
    write_to_sqldb,
)

__version__ = "2025.6.30"

filterwarnings("ignore", "Not using MPI")
filterwarnings("ignore", "Unexpected warning from scipy")
filterwarnings("ignore", "using slow exponentiator")
filterwarnings("ignore", ".*creased to keep within bounds")
filterwarnings("ignore", "Used mean of.*", module="cogent3")
filterwarnings("ignore", "use.*")
filterwarnings("ignore", category=DeprecationWarning)


def get_opt_settings(testrun):
    """create optimisation settings."""
    return (
        {"max_restarts": 1, "limit_action": "ignore", "max_evaluations": 10}
        if testrun
        else None
    )


class OrderedGroup(click.Group):
    """custom group class to ensure help function returns commands in desired order.
    class is adapted from Максим Стукало's answer to
    https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    """

    def __init__(
        self,
        name: str | None = None,
        commands: Mapping[str, click.Command] | None = None,
        **kwargs,
    ):
        super().__init__(name, commands, **kwargs)
        #: the registered subcommands by their exported names.
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@trogon.tui()
@click.group(cls=OrderedGroup)
@click.version_option(__version__)
def main():
    """mdeq: mutation disequilibrium analysis tools."""


@main.command(no_args_is_help=True)
@_cli_opt._indir
@_cli_opt._suffix
@_cli_opt._inpath
@_cli_opt._outpath
@_cli_opt._min_length
@_cli_opt._fg_edge
@_cli_opt._codon_pos
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
def prep(
    indir,
    suffix,
    inpath,
    outpath,
    min_length,
    fg_edge,
    codon_pos,
    limit,
    overwrite,
    verbose,
):
    """pre-process alignment data.

    Gaps and degenerate nucleotide characters will always be removed and
    alignments < `--min_length` will be excluded.
    """
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_args()
    console = Console()
    if indir and inpath:
        console.print(
            r"[red]EXIT: ambiguous input, set either --indir [b]AND[/b] --suffix [b]OR[/b] --inpath",
        )
        sys.exit(1)

    if indir and suffix is None:
        console.print(r"[red]EXIT: must define suffix")
        sys.exit(1)

    dstore = open_data_store(inpath or indir, suffix=suffix, limit=limit)
    loader = (
        load_from_sqldb()
        if inpath
        else get_app("load_aligned", format_name=suffix, moltype="dna")
    )
    if fg_edge:
        aln = loader(dstore.completed[0])
        if fg_edge not in aln.names:
            console.print(rf"[red]EXIT: {fg_edge=} not in {aln.names}")
            sys.exit(1)

    LOGGER.log_file_path = f"{outpath.stem}-prep.log"

    app_series = [loader]
    if codon_pos:
        app_series.append(get_app("take_codon_positions", int(codon_pos)))

    app_series.extend(
        [get_app("omit_degenerates", moltype="dna"), get_app("min_length", min_length)],
    )
    if fg_edge:
        app_series.append(set_fg_edge(fg_edge=fg_edge))

    if overwrite:
        Path(outpath).unlink(missing_ok=True)

    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    app_series.append(write_to_sqldb(out_dstore))

    app = reduce(add, app_series)
    app.apply_to(
        dstore.completed,
        logger=LOGGER,
        cleanup=True,
        show_progress=verbose > 0,
    )
    console.print("[green]Done!")


@main.command(no_args_is_help=True)
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
    dstore = open_data_store(inpath, limit=limit)
    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)

    sample_ids = {m.unique_id.replace(".json", "") for m in dstore.completed}
    paired = physically_adjacent(gene_order, sample_ids)
    # make the grouped data app
    group_loader = load_data_group(inpath)
    for pair in track(paired, refresh_per_second=5):
        record = group_loader(pair)
        writer(record)

    log_file_path = Path(LOGGER.log_file_path)
    LOGGER.shutdown()
    writer.data_store.write_log(
        unique_id=log_file_path.name,
        data=log_file_path.read_text(),
    )
    log_file_path.unlink()
    writer.data_store.close()

    writer.data_store.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command(no_args_is_help=True)
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
    # TODO need a separate command to apply foreground_from_jsd() to an
    #  alignment for decorating alignments with the foreground edge
    # or check alignment.info for a fg_edge key -- all synthetic data
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-toe.log"
    LOGGER.log_args()

    dstore = open_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if not matches_type(dstore, expected_types):
        click.secho(
            f"records {dstore.record_type} not one of the expected types {expected_types}",
            fg="red",
        )
        sys.exit(1)

    loader = load_from_sqldb()

    # check consistency of just_continuous / fg_edge / aln.info
    _aln = loader(dstore.completed[0])
    if just_continuous and fg_edge is not None:
        click.secho(
            f"WARN: setting just_continuous overrides {fg_edge!r} setting",
            fg="yellow",
        )

    # if fg_edge is specified then this value is checked for existence in alignment
    if fg_edge is not None:
        if fg_edge not in _aln.names:
            click.secho(f"FAIL: {fg_edge!r} name not present in {_aln.names}", fg="red")
            sys.exit(1)

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
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)
    if inject_fg:
        app = loader + inject_fg + bstrapper + writer
    else:
        app = loader + bstrapper + writer

    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    if mpi:
        kwargs["par_kw"]["chunksize"] = 1
    app.apply_to(
        dstore.completed,
        logger=LOGGER,
        cleanup=True,
        show_progress=verbose > 2,
        **kwargs,
    )
    out_dstore.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command(no_args_is_help=True)
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

    dstore = open_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if not matches_type(dstore, expected_types):
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        sys.exit(1)

    # construct hypothesis app, null constrains edge_names to same process
    loader = load_from_sqldb()
    opt_args = get_opt_settings(testrun)
    teop = temporal_eop(edge_names, tree=treepath, opt_args=opt_args)
    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)
    process = loader + teop + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    process.apply_to(
        dstore.completed,
        logger=LOGGER,
        cleanup=True,
        show_progress=verbose > 2,
        **kwargs,
    )
    out_dstore.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command(no_args_is_help=True)
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

    dstore = open_data_store(inpath, limit=limit)
    expected_types = ("grouped",)
    if not matches_type(dstore, expected_types):
        click.secho(
            f"records {dstore.record_type} not one of the expected types {expected_types}",
            fg="red",
        )
        sys.exit(1)

    loader = load_from_sqldb()
    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)
    test_adjacent = adjacent_eop(
        tree=treepath,
        opt_args=get_opt_settings(testrun),
        share_mprobs=share_mprobs,
    )
    process = loader + test_adjacent + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    _ = process.apply_to(
        dstore.completed,
        logger=LOGGER,
        cleanup=True,
        show_progress=verbose > 1,
        **kwargs,
    )
    out_dstore.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command(no_args_is_help=True)
@_cli_opt._inpath_bootstrap
@_cli_opt._outpath
@_cli_opt._wrt_nstat
@_cli_opt._parallel
@_cli_opt._mpi
@_cli_opt._limit
@_cli_opt._overwrite
@_cli_opt._verbose
def convergence(inpath, outpath, wrt_nstat, parallel, mpi, limit, overwrite, verbose):
    """estimates convergence towards mutation equilibrium."""
    with keep_running():
        LOGGER = CachingLogger(create_dir=True)
        LOGGER.log_file_path = f"{outpath.stem}-mdeq-convergence.log"
        LOGGER.log_args()
        dstore = open_data_store(inpath, limit=limit)
        expected_types = ("compact_bootstrap_result",)
        if not matches_type(dstore, expected_types):
            click.secho(
                f"records {dstore.record_type} not one of the expected types {expected_types}",
                fg="red",
            )
            sys.exit(1)

        loader = load_from_sqldb()
        to_delta_nabla = bootstrap_to_nabla(wrt_nstat=wrt_nstat)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if outpath.exists() and overwrite:
            outpath.unlink()
        out_dstore = open_data_store(outpath, mode="w")
        writer = write_to_sqldb(out_dstore)
        process = loader + to_delta_nabla + writer
        kwargs = configure_parallel(parallel=parallel, mpi=mpi)
        process.apply_to(
            dstore.completed,
            logger=LOGGER,
            cleanup=True,
            show_progress=verbose > 1,
            **kwargs,
        )
        out_dstore.close()

        func_name = inspect.stack()[0].function
        click.secho(f"{func_name!r} is done!", fg="green")


@main.command(no_args_is_help=True)
@_cli_opt._inpath_controls
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
    dstore = open_data_store(inpath, limit=limit)
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
    console = Console()
    if not matches_type(dstore, (result_types[analysis],)):
        console.print(
            "[red]ERROR: "
            f"object type in {inpath!r} does not match expected "
            f"{result_types[analysis]!r} for analysis {analysis!r}",
        )
        sys.exit(1)

    model_name = control_name[analysis][controls]
    model_selector = select_model_result(model_name)

    loader = load_from_sqldb()
    generator = control_generator(model_selector, seed=seed)

    # now use that rng to randomly select sample_size from dstore
    if verbose:
        print(f"{dstore!r}")

    if sample_size is not None:
        # sample without replacement
        dstore = [
            dstore.completed[i]
            for i in generator.rng.sample(range(len(dstore.completed)), sample_size)
        ]
        if verbose > 3:
            print(f"{dstore!r}")

    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)
    proc = loader + generator + writer
    proc.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    out_dstore.close()
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


# TODO postprocess functions, generate figures, tabulate data
@main.command(no_args_is_help=True)
@_cli_opt._inpath
def db_summary(inpath):
    """displays summary information about a db"""
    db_status(inpath)


@main.command(no_args_is_help=True)
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

    reader = load_from_sqldb()
    paths = paths_to_sqlitedbs_matching(indir, pattern, recursive)
    if not paths:
        console.print(
            f"[red]EXIT: no paths found for {indir=}, {recursive=!r}, {pattern=!r}",
        )
        sys.exit(1)

    with Progress(transient=True) as progress:
        all_paths = progress.add_task("[green]Dbs...", total=len(paths))
        for i, path in enumerate(paths):
            progress.update(all_paths, completed=i + 1)
            if outdir:
                outpath = outdir / f"{path.stem}.tsv"
            else:
                outpath = path.parent / f"{path.stem}.tsv"

            if outpath.exists() and overwrite:
                outpath.unlink()
            elif outpath.exists() and not overwrite:
                if verbose:
                    console.print(f"[green]{outpath} exists, skipping")
                continue

            dstore = open_data_store(path, limit=limit)
            if not matches_type(dstore, data_type):
                if verbose:
                    console.print(
                        "[yellow]SKIPPED: "
                        f"record type {dstore.record_type!r} in '{path}' does not match "
                        f"expected {data_type!r}",
                    )
                continue

            records = progress.add_task(
                "[blue]records...",
                total=len(dstore.completed),
                transient=True,
            )

            data = defaultdict(list)
            for j, m in enumerate(dstore.completed):
                progress.update(records, completed=j + 1)
                r = reader(m)
                if r.observed.pvalue is None:
                    continue

                data["name"].append(m.unique_id)
                data["chisq_pval"].append(r.observed.pvalue)
                data["bootstrap_pval"].append(r.pvalue)

            progress.remove_task(records)

            table = make_table(data=data)
            table.write(outpath)
    console.print("[green]Done!")


@main.command(no_args_is_help=True)
@_cli_opt._inpath_convergence
@_cli_opt._outdir
@_cli_opt._overwrite
@_cli_opt._verbose
def extract_nabla_c(inpath, outdir, overwrite, verbose):
    """extracts nabla-c from convergence sqlitedb results

    generates a tsv with same name in outdir"""
    console = Console()
    if verbose:
        console.print(f"[blue]{inpath!s}")

    outpath = outdir / inpath.with_suffix(".tsv")
    if overwrite:
        outpath.unlink(missing_ok=True)
    elif outpath.exists():
        console.print(f"[red]{outpath!s} exists, set overwrite to replace")
        sys.exit(1)

    dstore = open_data_store(inpath, mode="r")
    data_type = "delta_nabla", "nabla_c"
    console = Console()
    if not matches_type(dstore, data_type):
        console.print(
            "[red]ERROR: "
            f"object type in {inpath!r} does not match expected "
            f"{data_type!r}",
        )
        sys.exit(1)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    table = nabla_c_table(dstore)
    table.write(outpath)
    console.print("[green]Done!")


@main.command(no_args_is_help=True)
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
            f"[red] {window_size=} is less than {min_length=}, should be other way around",
        )
        sys.exit(1)

    dstore = open_data_store(inpath)
    expected_types = ("ArrayAlignment", "Alignment")
    if not matches_type(dstore, expected_types):
        click.secho(
            f"records {dstore.record_type} not one of the expected types {expected_types}",
            fg="red",
        )
        sys.exit(1)

    loader = load_from_sqldb()
    if outpath.exists() and overwrite:
        outpath.unlink(missing_ok=True)
    out_dstore = open_data_store(outpath, mode="w" if overwrite else "r")
    writer = write_to_sqldb(out_dstore)
    with Progress() as progress:
        alignments = progress.add_task(
            "[green]Alignment...",
            total=len(dstore.completed),
        )
        for i, member in enumerate(dstore.completed):
            progress.update(alignments, completed=i + 1)
            n = omit_suffixes_from_path(Path(member.unique_id))
            aln = loader(member)
            num_windows = len(aln) - window_size + 1
            windows = progress.add_task(
                "[blue]slide...",
                total=num_windows,
                transient=True,
            )
            for start in range(0, num_windows, step):
                progress.update(windows, completed=start + 1)
                sliced = aln[start : start + window_size]
                sub = sliced.no_degenerates()
                if sub is None or len(sub) < min_length:
                    continue

                sub.source = f"{n}-{start}"
                sub.info.index = start
                if "fg_edge" in aln.info:
                    sub.info.fg_edge = aln.info["fg_edge"]

                # write it
                writer(sub)
            progress.remove_task(windows)

    log_file_path = Path(LOGGER.log_file_path)
    LOGGER.shutdown()
    writer.data_store.write_log(
        unique_id=log_file_path.name,
        data=log_file_path.read_text(),
    )
    log_file_path.unlink()
    out_dstore.close()
    console.print("[green]Done!")


if __name__ == "__main__":
    main()
