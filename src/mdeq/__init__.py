"""mdeq: mutation disequilibrium analysis tools."""

# following line to stop automatic threading by numpy

from mdeq import _block_threading  # isort: skip  # make sure this stays at the top
import pathlib

from warnings import filterwarnings

import click

from cogent3 import load_tree
from cogent3.app import io
from scitrack import CachingLogger

from mdeq import adjacent as _adjacent
from mdeq import (
    model as _model,  # to ensure registration of define substitution models
)
from mdeq.bootstrap import bootstrap_toe
from mdeq.control import select_model_result
from mdeq.convergence import bootstrap_to_nabla
from mdeq.eop import ALT_AEOP, ALT_TEOP, NULL_AEOP, NULL_TEOP
from mdeq.lrt import ALT_TOE, NULL_TOE
from mdeq.utils import get_obj_type


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

__version__ = "2021.12.20"

filterwarnings("ignore", "Not using MPI")
filterwarnings("ignore", "Unexpected warning from scipy")
filterwarnings("ignore", "using slow exponentiator")
filterwarnings("ignore", ".*creased to keep within bounds")
filterwarnings("ignore", "Used mean of.*", module="cogent3")


def get_opt_settings(testrun):
    """create optimisation settings."""
    return (
        {"max_restarts": 1, "limit_action": "ignore", "max_evaluations": 10}
        if testrun
        else None
    )


def _process_comma_seq(*args):
    val = args[-1]
    return val.split(",") if val else val


def _gene_order_table(*args):
    """returns a cogent3 Table with required columns.

    Raises
    ------
    ValueError if required column names are not present
    """
    from cogent3 import load_table

    table = load_table(args[-1])
    required = {"name", "coord_name", "start"}
    missing = required - set(table.header)
    if missing:
        raise ValueError(f"missing {missing!r} columns from gene order table")

    return table[:, list(required)]


def _valid_path(path, exists):
    path = pathlib.Path(path)
    if exists and not path.exists():
        raise ValueError(f"{path!r} does not exist")

    if path.suffix != ".tinydb":
        raise ValueError(f"{path!r} is not a tinydb")
    return path


def _valid_tinydb_input(*args):
    # input path must exist!
    return _valid_path(args[-1], True)


def _valid_tinydb_output(*args):
    return _valid_path(args[-1], False)


def _load_tree(*args):
    path = args[-1]
    return load_tree(path) if path else path


@click.group()
@click.version_option(__version__)
def main():
    """mdeq: mutation disequilibrium analysis tools."""
    pass


_inpath = click.option(
    "-i", "--inpath", callback=_valid_tinydb_input, help="path to a tinydb of aligments"
)
_outpath = click.option(
    "-o",
    "--outpath",
    callback=_valid_tinydb_output,
    help="path to create a result tinydb",
)
_treepath = click.option(
    "-T",
    "--treepath",
    callback=_load_tree,
    help="path to newick formatted phylogenetic tree",
)
_num_reps = click.option(
    "-n", "num_reps", type=int, default=100, help="number of samples to simulate"
)
_seed = click.option("-s", "--seed", type=int, help="seed for random number generator")
_verbose = click.option("-v", "--verbose", count=True)
_limit = click.option("-L", "--limit", type=int, default=None)
_overwrite = click.option("-O", "--overwrite", is_flag=True)
_testrun = click.option(
    "-t",
    "--testrun",
    is_flag=True,
    help="don't write anything, quick (but inaccurate) optimisation",
)
_fg_edge = click.option(
    "-fg", "--foreground_edge", help="foreground edge to test for equilibrium"
)
_bg_edge = click.option(
    "-bg",
    "--background_edges",
    callback=_process_comma_seq,
    help="apply discrete-time process to these edges",
)


@main.command()
@_inpath
@click.option(
    "-g",
    "--gene_order",
    required=True,
    callback=_gene_order_table,
    help="path to gene order table, note must contain"
    " 'name', 'coord_name' and 'start' columns",
)
@_outpath
@_limit
@_overwrite
@_verbose
@_testrun
def make_adjacent(inpath, gene_order, outpath, limit, overwrite, verbose, testrun):
    """makes tinydb of adjacent alignment records."""
    from tqdm import tqdm

    from .adjacent import load_data_group, physically_adjacent

    LOGGER = CachingLogger(create_dir=True)

    LOGGER.log_file_path = outpath.parent / "mdeq-make_adjacent.log"
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

    writer.data_store.close()
    click.secho("Done!", fg="green")


@main.command()
@_inpath
@_treepath
@_outpath
@_bg_edge
@_num_reps
@_limit
@_overwrite
@_verbose
@_testrun
def toe(
    inpath,
    treepath,
    outpath,
    background_edges,
    num_reps,
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
    LOGGER.log_file_path = outpath.parent / "mdeq-toe.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records one of the expected types {expected_types}", fg="red")
        exit()

    loader = io.load_db()
    opt_args = get_opt_settings(testrun)
    bstrapper = bootstrap_toe(tree=treepath, num_reps=num_reps, opt_args=opt_args)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + bstrapper + writer
    process.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    click.secho("Done!", fg="green")


@main.command()
@_inpath
@_outpath
@_treepath
@click.option(
    "-e",
    "--edge_names",
    callback=_process_comma_seq,
    required=True,
    help="comma separated edge names to test for equivalence",
)
@_limit
@_overwrite
@_verbose
@_testrun
def teop(inpath, outpath, treepath, edge_names, limit, overwrite, verbose, testrun):
    """test of equivalence of mutation equilibrium between branches."""
    from .eop import adjacent_eop, temporal_eop

    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = outpath.parent / "mdeq-teop.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    # construct hypothesis app, null constrains edge_names to same process
    loader = io.load_db()
    opt_args = get_opt_settings(testrun)
    teop = temporal_eop(edge_names, tree=treepath, opt_args=opt_args)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + teop + writer
    process.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    click.secho("Done!", fg="green")


# todo add args for parallelisation
@main.command()
@_inpath
@_outpath
@_treepath
@click.option(
    "-s",
    "--share_mprobs",
    is_flag=True,
    help="constrain loci to have the same motif probs",
)
@_limit
@_overwrite
@_verbose
@_testrun
def aeop(inpath, outpath, treepath, share_mprobs, limit, overwrite, verbose, testrun):
    """test of equivalence of mutation equilibrium between adjacent loci."""
    from .adjacent import load_data_group, physically_adjacent
    from .eop import adjacent_eop

    LOGGER = CachingLogger(create_dir=True)

    LOGGER.log_file_path = outpath.parent / "mdeq-aeop.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    loader = io.load_db()
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    test_adjacent = adjacent_eop(
        tree=treepath, opt_args=get_opt_settings(testrun), share_mprobs=share_mprobs
    )
    process = loader + test_adjacent + writer
    _ = process.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 1)
    click.secho("Done1", fg="green")


@main.command()
@_inpath
@_outpath
@_limit
@_overwrite
@_verbose
def convergence(inpath, outpath, limit, overwrite, verbose):
    """uses output from toe to generate delta_nabla."""
    LOGGER = CachingLogger(create_dir=True)
    LOGGER.log_file_path = outpath.parent / "mdeq-convergence.log"
    LOGGER.log_args()
    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("compact_bootstrap_result",)
    if get_obj_type(dstore) not in expected_types:
        click.secho(f"records one of the expected types {expected_types}", fg="red")
        exit()

    loader = io.load_db()
    to_delta_nabla = bootstrap_to_nabla()
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + to_delta_nabla + writer
    r = process.apply_to(dstore, logger=True, cleanup=True, show_progress=verbose > 1)


@main.command()
@_inpath
@_outpath
@click.option(
    "-a",
    "--analysis",
    type=click.Choice(["aeop", "teop", "toe", "single-model"]),
    required=True,
    help="which control set to generate",
)
@click.option(
    "--controls",
    type=click.Choice(["-ve", "+ve"]),
    required=True,
    help="which control set to generate",
)
@_limit
@_overwrite
@_verbose
@_testrun
def make_controls(
    inpath, outpath, analysis, controls, limit, overwrite, verbose, testrun
):
    """simulate negative and positive controls

    Notes
    -----
    A single simulated record is produced for each input record.
    """
    LOGGER = CachingLogger(create_dir=True)
    # todo support specifying the rng seed
    LOGGER.log_file_path = outpath.parent / "mdeq-make_controls.log"
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
        exit()

    model_name = control_name[analysis][controls]
    model_selector = select_model_result(model_name)

    loader = io.load_db()
    generator = control.control_generator(model_selector)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    proc = loader + generator + writer
    proc.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    click.secho("Done!", fg="green")


# todo postprocess functions, generate figures, tabulate data

if __name__ == "__main__":
    main()
