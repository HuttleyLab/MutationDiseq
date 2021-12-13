import os
from tempfile import TemporaryDirectory

import pytest
from cogent3 import load_aligned_seqs
from cogent3.app import io
from cogent3.app.result import generic_result

from kath_library.bootstrap import (
    confidence_interval,
    create_bootstrap_app,
    create_bootstrap_app_diff_trees,
    estimate_pval,
)
from kath_library.convergence import _get_convergence
from kath_library.t50 import _get_t50


@pytest.fixture()
def aln():
    aln = load_aligned_seqs("~/repos/cogent3/tests/data/brca1.fasta", moltype="dna")
    aln = aln.take_seqs(["TombBat", "RoundEare", "DogFaced"])
    aln = aln.no_degenerates(motif_length=3)
    aln.info["fg_edge"] = "TombBat"

    return aln


@pytest.fixture()
def dstore_instance():
    return io.get_data_store(
        "~/repos/data/microbial/synthetic/758_443154_73021/3000bp.tinydb"
    )


def test_create_bootstrap_app(aln):
    bstrap = create_bootstrap_app(2)
    bootstrap = bstrap(aln)

    assert isinstance(bootstrap, generic_result)
    assert len(bootstrap) == 3


def test_create_bootstrap_app_composable(dstore_instance):
    with TemporaryDirectory(dir=".") as dirname:
        reader = io.load_db()
        outpath = os.path.join(os.getcwd(), dirname, "tempdir.tinydb")
        writer = io.write_db(outpath)
        bstrap = create_bootstrap_app(2)
        process = reader + bstrap + writer

        process.apply_to(dstore_instance[:1])
        assert len(process.data_store.summary_incomplete) == 0


def test_confidence_interval_with_convergence(aln):
    get_conf_int = confidence_interval(_get_convergence, 1)
    c_int = get_conf_int.run(aln)

    assert isinstance(c_int["observed"]["convergence"], float)
    assert isinstance(c_int["sim_1-result"]["convergence"], float)


def test_confidence_interval_with_t50(aln):
    get_conf_int = confidence_interval(_get_t50, 1)
    c_int = get_conf_int.run(aln)

    assert isinstance(c_int["observed"]["T50"], float)
    assert isinstance(c_int["sim_1-result"]["T50"], float)


def test_confidence_interval_app_composable(dstore_instance):
    with TemporaryDirectory(dir=".") as dirname:
        reader = io.load_db()
        outpath = os.path.join(os.getcwd(), dirname, "tempdir.tinydb")
        writer = io.write_db(outpath)
        c_int = confidence_interval(_get_t50, 1)
        process = reader + c_int + writer

        process.apply_to(dstore_instance[:1])
        assert len(process.data_store.summary_incomplete) == 0


def test_estimate_pval(aln):
    bstrap = create_bootstrap_app(2, discrete_edges=["443154", "73021"])
    bootstrap = bstrap(aln)
    assert estimate_pval(bootstrap) == 0

    bootstrap[3] = 100
    assert estimate_pval(bootstrap) == 1 / 3

    bootstrap[4] = 100
    assert estimate_pval(bootstrap) == 1 / 2
