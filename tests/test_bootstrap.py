import pathlib

import pytest

from cogent3 import load_aligned_seqs
from cogent3.app import io
from cogent3.util.deserialise import deserialise_object

from mdeq.bootstrap import create_bootstrap_app, estimate_pval


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def opt_args():
    return {"max_evaluations": 100, "limit_action": "ignore"}


@pytest.fixture()
def aln():
    aln = load_aligned_seqs(DATADIR / "brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)
    aln.info["fg_edge"] = "TombBat"

    return aln


@pytest.fixture()
def dstore_instance():
    return io.get_data_store(DATADIR / "3000bp.tinydb")


def test_create_bootstrap_app(aln, opt_args):
    bstrap = create_bootstrap_app(1, opt_args=opt_args)
    bootstrap = bstrap(aln)

    # assert isinstance(bootstrap, generic_result)
    assert len(bootstrap) == 2
    rd = bootstrap.to_rich_dict()


def test_deserialise_compact_boostrap_result(aln, opt_args):
    import json

    bstrap = create_bootstrap_app(1, opt_args=opt_args)
    result = bstrap(aln)
    txt = result.to_json()
    d = json.loads(txt)
    got = deserialise_object(txt)

    assert len(result) == 2


def test_create_bootstrap_app_composable(tmp_path, dstore_instance, opt_args):
    reader = io.load_db()
    outpath = tmp_path / "tempdir.tinydb"
    writer = io.write_db(outpath)
    bstrap = create_bootstrap_app(2, opt_args=opt_args)
    process = reader + bstrap + writer

    process.apply_to(dstore_instance[:1])
    assert len(process.data_store.summary_incomplete) == 0


def test_estimate_pval(aln, opt_args):
    opt_args["max_evaluations"] = 2000
    bstrap = create_bootstrap_app(
        2, discrete_edges=["443154", "73021"], opt_args=opt_args
    )
    bootstrap = bstrap(aln)
    assert estimate_pval(bootstrap) == 0

# todo write tests that don't specify a fg edge; that have > 3 taxa and specify a tree