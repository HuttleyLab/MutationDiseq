"""mdeq: mutation disequilibrium analysis tools."""

# following line to stop automatic threading by numpy
import inspect
import sys

from warnings import filterwarnings

import click

from cogent3.app import io
from scitrack import CachingLogger
from tqdm import tqdm

from mdeq import (
    model as _model,  # required to ensure registration of define substitution models
)
from mdeq._click_options import (
    _analysis,
    _controls,
    _edge_names,
    _fg_edge,
    _gene_order,
    _gene_order_table,
    _inpath,
    _just_continuous,
    _limit,
    _mpi,
    _num_reps,
    _outpath,
    _overwrite,
    _parallel,
    _process_comma_seq,
    _seed,
    _sequential,
    _share_mprobs,
    _testrun,
    _treepath,
    _verbose,
    _wrt_nstat,
)
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
from mdeq.toe import ALT_TOE, NULL_TOE
from mdeq.utils import configure_parallel, get_obj_type, set_fg_edge


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
@_inpath
@_gene_order
@_outpath
@_limit
@_overwrite
@_verbose
@_testrun
def make_adjacent(inpath, gene_order, outpath, limit, overwrite, verbose, testrun):
    """makes tinydb of adjacent alignment records."""
    LOGGER = CachingLogger(create_dir=True)

    LOGGER.log_file_path = outpath.parent / f"{outpath.stem}-mdeq-make_adjacent.log"
    LOGGER.log_args()

    # we get member names from input dstore
    dstore = io.get_data_store(inpath, limit=limit)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )

    sample_ids = {m.name.replace(".json", "") for m in dstore}
    paired = physically_adjacent(gene_order, sample_ids)
    # make the grouped data app
    group_loader = load_data_group(inpath)
    for pair in tqdm(paired):
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
@_inpath
@_treepath
@_outpath
@_just_continuous
@_fg_edge
@_sequential
@_num_reps
@_parallel
@_mpi
@_limit
@_overwrite
@_verbose
@_testrun
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

    loader = io.load_db()

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
    writer = io.write_db(
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
@_inpath
@_outpath
@_treepath
@_edge_names
@_parallel
@_mpi
@_limit
@_overwrite
@_verbose
@_testrun
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
    loader = io.load_db()
    opt_args = get_opt_settings(testrun)
    teop = temporal_eop(edge_names, tree=treepath, opt_args=opt_args)
    writer = io.write_db(
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
@_inpath
@_outpath
@_treepath
@_share_mprobs
@_parallel
@_mpi
@_limit
@_overwrite
@_verbose
@_testrun
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

    loader = io.load_db()
    writer = io.write_db(
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
@_inpath
@_outpath
@_wrt_nstat
@_parallel
@_mpi
@_limit
@_overwrite
@_verbose
def convergence(inpath, outpath, wrt_nstat, parallel, mpi, limit, overwrite, verbose):
    """estimates convergence towards mutation equilibrium."""
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = f"{outpath.stem}-mdeq-convergence.log"
    LOGGER.log_args()
    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("compact_bootstrap_result",)
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records not one of the expected types {expected_types}", fg="red")
        exit(1)

    loader = io.load_db()
    to_delta_nabla = bootstrap_to_nabla(wrt_nstat=wrt_nstat)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + to_delta_nabla + writer
    kwargs = configure_parallel(parallel=parallel, mpi=mpi)
    r = process.apply_to(
        dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 1, **kwargs
    )
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


@main.command()
@_inpath
@_outpath
@_analysis
@_controls
@_seed
@_limit
@_overwrite
@_verbose
@_testrun
def make_controls(
    inpath, outpath, analysis, controls, seed, limit, overwrite, verbose, testrun
):
    """simulate negative and positive controls

    Notes
    -----
    A single simulated record is produced for each input record.
    """
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = outpath.parent / f"{outpath.stem}-mdeq-make_controls.log"
    LOGGER.log_args()
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

    loader = io.load_db()
    generator = control_generator(model_selector, seed=seed)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    proc = loader + generator + writer
    proc.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    func_name = inspect.stack()[0].function
    click.secho(f"{func_name!r} is done!", fg="green")


# todo postprocess functions, generate figures, tabulate data

if __name__ == "__main__":
    main()
