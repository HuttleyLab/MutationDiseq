import pathlib

import click

from cogent3 import load_table, load_tree


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]


def _rand_seed(*args):
    """handles random seed input"""
    import time

    return int(args[-1]) if args[-1] else int(time.time())


def _process_comma_seq(*args):
    val = args[-1]
    return val.split(",") if val else val


def _gene_order_table(*args):
    """returns a cogent3 Table with required columns.

    Raises
    ------
    ValueError if required column names are not present
    """
    table = load_table(args[-1])
    required = {"name", "coord_name", "start"}
    if missing := required - set(table.header):
        raise ValueError(f"missing {missing!r} columns from gene order table")

    return table[:, list(required)]


def _valid_path(path, must_exist):
    path = pathlib.Path(path)
    if must_exist and not path.exists():
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
_seed = click.option(
    "-s",
    "--seed",
    callback=_rand_seed,
    default=None,
    help="seed for random number generator, defaults to system clock",
)
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
_mpi = click.option(
    "-m", "--mpi", type=int, default=0, help="use MPI with this number of procs"
)
_parallel = click.option(
    "-p",
    "--parallel",
    is_flag=True,
    help="run in parallel (on single machine)",
)
_gene_order = click.option(
    "-g",
    "--gene_order",
    required=True,
    callback=_gene_order_table,
    help="path to gene order table, note must contain"
    " 'name', 'coord_name' and 'start' columns",
)
_edge_names = click.option(
    "-e",
    "--edge_names",
    callback=_process_comma_seq,
    required=True,
    help="comma separated edge names to test for equivalence",
)
_share_mprobs = click.option(
    "-m",
    "--share_mprobs",
    is_flag=True,
    help="constrain loci to have the same motif probs",
)
_analysis = click.option(
    "-a",
    "--analysis",
    type=click.Choice(["aeop", "teop", "toe", "single-model"]),
    required=True,
    help="which control set to generate",
)
_controls = click.option(
    "--controls",
    type=click.Choice(["-ve", "+ve"]),
    required=True,
    help="which control set to generate",
)
