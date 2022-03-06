"""mdeq: mutation disequilibrium analysis tools"""

# following line to stop automatic threading by numpy
from . import _block_threading  # isort: skip  # make sure this stays at the top
import json
import pathlib

from warnings import filterwarnings

import click

from cogent3.app import io
from scitrack import CachingLogger

from mdeq.bootstrap import bootstrap_toe
from mdeq.convergence import bootstrap_to_nabla

from . import (
    model as _model,  # to ensure registration of define substitution models
)


__version__ = "2021.12.20"

filterwarnings("ignore", "Not using MPI")
filterwarnings("ignore", "Unexpected warning from scipy")
filterwarnings("ignore", "using slow exponentiator")
filterwarnings("ignore", ".*decreased to keep within bounds")


LOGGER = CachingLogger(create_dir=True)


def _process_comma_seq(*args):
    return args[-1].split(",")


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


def valid_result_types(dstore, types):
    """fail if the record type in dstore is not within types."""
    from cogent3.app import data_store

    data = json.loads(dstore[0].read())
    type_ = data["type"].split(".")[-1]
    return type_ in types


@click.group()
@click.version_option(__version__)
def main():
    """mdeq: mutation disequilibrium analysis tools"""
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
_num_reps = click.option(
    "-n", "num_reps", type=int, default=100, help="number of samples to simulate"
)
_seed = click.option("-s", "--seed", type=int, help="seed for random number generator")
_verbose = click.option("-v", "--verbose", count=True)
_limit = click.option("-L", "--limit", type=int, default=None)
_overwrite = click.option("-O", "--overwrite", is_flag=True)
_testrun = click.option("-t", "--testrun", is_flag=True, help="don't write anything")
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
@_outpath
@_bg_edge
@_num_reps
@_limit
@_overwrite
@_verbose
def toe(
    inpath,
    outpath,
    background_edges,
    num_reps,
    limit,
    overwrite,
    verbose,
):
    """test of existence of mutation equilibrium."""
    from cogent3.app import io

    # or check alignment.info for a fg_edge key -- all synthetic data
    LOGGER.log_file_path = outpath.parent / "mdeq-toe.log"
    LOGGER.log_args()

    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("ArrayAlignment", "Alignment")
    if not valid_result_types(dstore, expected_types):
        click.secho(f"records one of the expected types {expected_types}", fg="red")
        exit()

    loader = io.load_db()
    bstrapper = create_bootstrap_app(num_reps=num_reps, discrete_edges=background_edges)
    writer = io.write_db(
        outpath, create=True, if_exists="overwrite" if overwrite else "raise"
    )
    process = loader + bstrapper + writer
    process.apply_to(dstore, logger=LOGGER, cleanup=True, show_progress=verbose > 2)
    click.secho("Done!", fg="green")


@main.command()
@_inpath
@_outpath
@click.option("-e", "--edge_names", required=True, help="edges to test for equivalence")
@_limit
@_overwrite
@_verbose
@_testrun
def teop(inpath, outpath, edge_names, limit, overwrite, verbose, testrun):
    """test of equivalence of mutation equilibrium between branches."""
    from .eop import adjacent_EOP, edge_EOP

    LOGGER.log_file_path = outpath.parent / "mdeq-teop.log"
    LOGGER.log_args()


# todo inoput for ordewred list of alignment names
@main.command()
@_inpath
@_outpath
@_limit
@_overwrite
@_verbose
@_testrun
def aeop(inpath, outpath, limit, overwrite, verbose, testrun):
    """test of equivalence of mutation equilibrium between loci."""
    from .eop import adjacent_EOP, edge_EOP

    LOGGER.log_file_path = outpath.parent / "mdeq-aeop.log"
    LOGGER.log_args()


@main.command()
@_inpath
@_outpath
@_num_reps
@_limit
@_overwrite
@_verbose
def convergence(inpath, outpath, limit, overwrite, verbose):
    """uses output from toe to generate delta_nabla."""
    LOGGER.log_file_path = outpath.parent / "mdeq-convergence.log"
    LOGGER.log_args()
    dstore = io.get_data_store(inpath, limit=limit)
    expected_types = ("compact_bootstrap_result",)
    if not valid_result_types(dstore, expected_types):
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
    "--controls",
    default=click.Choice(["-ve", "+ve"]),
    help="which control set to generate",
)
@_num_reps
@_limit
@_overwrite
@_verbose
@_testrun
def make_controls(
    inpath, outpath, controls, num_reps, limit, overwrite, verbose, testrun
):
    """simulate negative and positive controls.

    Note the input here MUST be hypothesis_result OR model_result in a
    tinydb
    """
    # todo the positive controls
    LOGGER.log_file_path = outpath.parent / "mdeq-make_controls.log"
    LOGGER.log_args()
    # create loader, read a single result and validate the type matches the controls choice
    # validate the model choice too


# todo postprocess functions, generate figures, tabulate data

if __name__ == "__main__":
    main()
