import pathlib

import pytest

from cogent3 import load_aligned_seqs, open_data_store
from cogent3.util.deserialise import deserialise_object

from mdeq.bootstrap import (
    bootstrap_toe,
    compact_bootstrap_result,
    create_bootstrap_app,
)
from mdeq.sqlite_data_store import load_from_sql, write_to_sqldb


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tinydb")


@pytest.fixture(autouse=True)
def workingdir(tmp_dir, monkeypatch):
    # this set's the working directory for all tests in this module
    # as a tmp dir
    monkeypatch.chdir(tmp_dir)


@pytest.fixture()
def opt_args():
    return {"max_evaluations": 100, "limit_action": "ignore", "max_restarts": 1}


@pytest.fixture()
def aln():
    aln = load_aligned_seqs(DATADIR / "brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)
    aln.info["fg_edge"] = "TombBat"

    return aln


@pytest.fixture()
def aligns_dstore():
    return open_data_store(DATADIR / "3000bp-new.sqlitedb")


@pytest.fixture()
def bstrap_result_dstore():
    return open_data_store(DATADIR / "fg_GSN_synthetic-lo_lo-300bp-1rep-new.sqlitedb")


def test_create_bootstrap_app(aln, opt_args):
    bstrap = create_bootstrap_app(num_reps=1, opt_args=opt_args)
    bootstrap = bstrap(aln)

    assert isinstance(bootstrap, compact_bootstrap_result)
    assert len(bootstrap) == 2
    rd = bootstrap.to_rich_dict()


def test_deserialise_compact_boostrap_result(aln, opt_args):
    bstrap = create_bootstrap_app(num_reps=1, opt_args=opt_args)
    result = bstrap(aln)
    txt = result.to_json()
    got = deserialise_object(txt)

    assert len(result) == 2
    assert isinstance(got, compact_bootstrap_result)

reader = load_from_sql()

def test_create_bootstrap_app_composable(tmp_path, aligns_dstore, opt_args):
    out_dstore = open_data_store(tmp_path / "tempdir-new.sqlitedb", mode="w")
    writer = write_to_sqldb(out_dstore)
    bstrap = create_bootstrap_app(num_reps=2, opt_args=opt_args)
    process = reader + bstrap + writer

    process.apply_to(aligns_dstore[:1], show_progress=False)
    assert len(out_dstore.summary_not_completed) == 0

    result = reader(out_dstore[0])
    assert isinstance(result, compact_bootstrap_result)
    pvalue = result.pvalue
    assert isinstance(pvalue, float)


def test_estimate_pval(bstrap_result_dstore):
    result = reader(bstrap_result_dstore[0])
    assert isinstance(result.pvalue, float)

    num_reps = sum(v.LR >= 0 for v in result.values()) - 1
    obs_lr = result.observed.LR
    num_ge_obs = (
        sum(result[k].LR >= obs_lr for k in result) - 1
    )  # adjust for comparing observed to itself
    got = result.pvalue
    assert got == num_ge_obs / num_reps


@pytest.fixture(scope="session")
def dstore4_tree():
    inpath = DATADIR / "4otu-aligns-new.sqlitedb"
    tree = "(Human,Platypus,(Mouse,Rat))"
    dstore = open_data_store(inpath)
    return dstore, tree


def num_discrete_edges(lf):
    from cogent3.recalculation.scope import InvalidScopeError

    num = 0
    for e in lf.tree.get_edge_vector(include_root=False):
        try:
            # is this edge modelled using discrete-time process
            lf.get_param_value("dpsubs", edge=e.name)
        except (InvalidScopeError, KeyError):
            pass
        else:
            num += 1
    return num


def test_4otu_create_bootstrap_app(dstore4_tree, opt_args):
    dstore, tree = dstore4_tree
    bstrap = create_bootstrap_app(
        tree=tree, num_reps=2, opt_args=opt_args, just_continuous=True
    )

    aln = reader(dstore[0])
    result = bstrap(aln)
    result.deserialised_values()
    assert isinstance(result, compact_bootstrap_result)
    n = num_discrete_edges(result.observed.null.lf)
    assert n == 0
    n = num_discrete_edges(result.observed.alt.lf)
    assert n == 0


def test_4otu_bootstrap_toe(dstore4_tree, opt_args):
    dstore, tree = dstore4_tree
    bstrap = bootstrap_toe(
        tree=tree, num_reps=2, opt_args=opt_args, just_continuous=True
    )

    aln = reader(dstore[0])
    result = bstrap(aln)
    result.deserialised_values()
    assert isinstance(result, compact_bootstrap_result)
    n = num_discrete_edges(result.observed["GSN"].lf)
    assert n == 0
    n = num_discrete_edges(result.observed["GN"].lf)
    assert n == 0
