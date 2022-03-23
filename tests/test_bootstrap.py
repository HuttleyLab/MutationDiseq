import pathlib

import pytest

from cogent3 import load_aligned_seqs
from cogent3.app import io
from cogent3.util.deserialise import deserialise_object

from mdeq.bootstrap import (
    bootstrap_toe,
    compact_bootstrap_result,
    create_bootstrap_app,
)


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


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
    return io.get_data_store(DATADIR / "3000bp.tinydb")


@pytest.fixture()
def bstrap_result_dstore():
    return io.get_data_store(DATADIR / "fg_GSN_synthetic-lo_lo-300bp-1rep.tinydb")


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


def test_create_bootstrap_app_composable(tmp_path, aligns_dstore, opt_args):
    reader = io.load_db()
    outpath = tmp_path / "tempdir.tinydb"
    writer = io.write_db(outpath)
    bstrap = create_bootstrap_app(num_reps=2, opt_args=opt_args)
    process = reader + bstrap + writer

    process.apply_to(aligns_dstore[:1])
    assert len(process.data_store.summary_incomplete) == 0
    writer.data_store.close()

    loader = io.load_db()
    dstore = io.get_data_store(outpath)
    result = loader(dstore[0])
    assert isinstance(result, compact_bootstrap_result)
    pvalue = result.pvalue
    assert isinstance(pvalue, float)


def test_estimate_pval(bstrap_result_dstore):
    loader = io.load_db()
    result = loader(bstrap_result_dstore[0])
    assert isinstance(result.pvalue, float)

    num_reps = sum(k != "observed" for k in result)
    obs_lr = result.observed.get_hypothesis_result("GSN", "GN").LR
    num_ge_obs = (
        sum(result[k].get_hypothesis_result("GSN", "GN").LR >= obs_lr for k in result)
        - 1
    )  # adjust for comparing observed to itself
    assert result.pvalue == num_ge_obs / num_reps


@pytest.fixture(scope="session")
def dstore4_tree():
    inpath = DATADIR / "4otu-aligns.tinydb"
    tree = "(Human,Platypus,(Mouse,Rat))"
    dstore = io.get_data_store(inpath)
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
    bstrap = create_bootstrap_app(tree=tree, num_reps=2, opt_args=opt_args)

    loader = io.load_db()
    aln = loader(dstore[0])
    result = bstrap(aln)
    assert isinstance(result, compact_bootstrap_result)
    n = num_discrete_edges(result.observed.null.lf)
    assert n == 0
    n = num_discrete_edges(result.observed.alt.lf)
    assert n == 0


def test_4otu_bootstrap_toe(dstore4_tree, opt_args):
    dstore, tree = dstore4_tree
    bstrap = bootstrap_toe(tree=tree, num_reps=2, opt_args=opt_args)

    loader = io.load_db()
    aln = loader(dstore[0])
    result = bstrap(aln)
    assert isinstance(result, compact_bootstrap_result)
    n = num_discrete_edges(result.observed["GSN"].lf)
    assert n == 0
    n = num_discrete_edges(result.observed["GN"].lf)
    assert n == 0
