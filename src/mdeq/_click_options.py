from pathlib import Path

import click
from cogent3 import load_table, load_tree


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
    if not path:
        return None

    path = Path(path)
    if must_exist and not path.exists():
        raise ValueError(f"{path!r} does not exist")

    if path.suffix != ".sqlitedb":
        raise ValueError(f"{path!r} is not a sqlitedb")
    return path


def _valid_sqlitedb_input(*args):
    # input path must exist!
    return _valid_path(args[-1], True)


def _valid_sqlitedb_output(*args):
    return _valid_path(args[-1], False)


def _load_tree(*args):
    path = args[-1]
    return load_tree(path) if path else path


_inpath = click.option(
    "-i",
    "--inpath",
    callback=_valid_sqlitedb_input,
    help="path to a sqlitedb of aligments",
)
_inpath_bootstrap = click.option(
    "-i",
    "--inpath",
    callback=_valid_sqlitedb_input,
    help="path to toe bootstrap result sqlitedb",
)
_inpath_convergence = click.option(
    "-i",
    "--inpath",
    callback=_valid_sqlitedb_input,
    help="path to convergence result sqlitedb",
)
_inpath_controls = click.option(
    "-i",
    "--inpath",
    callback=_valid_sqlitedb_input,
    help="path to sqlitedbs from any of the hypothesis tests",
)

_outpath = click.option(
    "-o",
    "--outpath",
    callback=_valid_sqlitedb_output,
    help="path to create a result sqlitedb",
)
_outdir = click.option(
    "-od",
    "--outdir",
    type=Path,
    help="directory to write output",
)
_treepath = click.option(
    "-T",
    "--treepath",
    callback=_load_tree,
    help="path to newick formatted phylogenetic tree",
)
_num_reps = click.option(
    "-n",
    "num_reps",
    type=int,
    default=100,
    help="number of samples to simulate",
)
_sample_size = click.option(
    "-z",
    "sample_size",
    type=int,
    default=None,
    help="number of observed alignments to sample, defaults to size of observed",
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
    "-fg",
    "--fg_edge",
    default=None,
    help="foreground edge to test for equilibrium",
)
_bg_edge = click.option(
    "-bg",
    "--bg_edges",
    callback=_process_comma_seq,
    help="apply discrete-time process to these edges",
)
_mpi = click.option(
    "-m",
    "--mpi",
    type=int,
    default=0,
    help="use MPI with this number of procs",
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
    "--share_mprobs",
    is_flag=True,
    help="constrain loci to have the same motif probs",
)
_analysis = click.option(
    "-a",
    "--analysis",
    type=click.Choice(["aeop", "teop", "toe", "single-model"]),
    required=True,
    help="analysis type",
)
_controls = click.option(
    "--controls",
    type=click.Choice(["-ve", "+ve"]),
    required=True,
    help="which control set to generate",
)
_just_continuous = click.option(
    "-J",
    "--just_continuous",
    is_flag=True,
    help="No discrete-time edges. Overrides any fg_edge settings in alignment.",
)
_sequential = click.option(
    "-q",
    "--sequential",
    is_flag=True,
    help="MLEs from simpler models are used to seed more complex ones",
)
_wrt_nstat = click.option(
    "-w",
    "--wrt_nstat",
    is_flag=True,
    help="nabla estimated using non-stationary calibrated Q. Otherwise, "
    "a stationary calibrated Q.",
)
_indir = click.option(
    "-id",
    "--indir",
    type=Path,
    help="path containing data files",
)
_pattern = click.option(
    "-g",
    "--pattern",
    default="",
    help="glob pattern for file names",
)
_recursive = click.option("-r", "--recursive", is_flag=True)
# for sliding window analysis
_window = click.option(
    "-wz",
    "window_size",
    type=int,
    default=600,
    help="size of slice",
    show_default=True,
)
_step = click.option(
    "-st",
    "step",
    type=int,
    default=10,
    help="step sizes between slices",
    show_default=True,
)
_min_length = click.option(
    "-ml",
    "--min_length",
    type=int,
    default=300,
    help="minimum length after filtering",
    show_default=True,
)

_suffix = click.option("-su", "--suffix", help="suffix of files to be loaded")
_codon_pos = click.option(
    "-c",
    "--codon_pos",
    type=click.Choice(["1", "2", "3"]),
    default=None,
    help="select this codon position, default is all.",
)
