import pathlib

import pytest
from cogent3 import make_aligned_seqs
from cogent3.app import io
from cogent3.app.evo import model_collection_result

from mdeq.lrt import (
    get_init_hypothesis,
    get_init_model_coll,
    get_lrt,
    get_no_init_hypothesis,
    get_no_init_model_coll,
)


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def dstore_instance():
    dstore = io.get_data_store(DATADIR / "3000bp.tinydb")
    return dstore


@pytest.fixture()
def mcr_dstore():
    return io.get_data_store(DATADIR / "mcr.tinydb")


@pytest.fixture()
def get_aln():
    _data = {
        "Human": "ATGCGGCTCGCGGAGGCCGCGCTCGCGGAG",
        "Mouse": "ATGCCCGGCGCCAAGGCAGCGCTGGCGGAG",
        "Opossum": "ATGCCAGTGAAAGTGGCGGCGGTGGCTGAG",
    }
    aln = make_aligned_seqs(data=_data, moltype="dna")
    aln.info["fg_edge"] = "Human"
    return aln


@pytest.fixture()
def get_aln_no_fg():
    _data = {
        "Human": "ATGCGGCTCGCGGAGGCCGCGCTCGCGGAG",
        "Mouse": "ATGCCCGGCGCCAAGGCAGCGCTGGCGGAG",
        "Opossum": "ATGCCAGTGAAAGTGGCGGCGGTGGCTGAG",
    }
    aln = make_aligned_seqs(data=_data, moltype="dna")

    return aln


@pytest.fixture()
def no_init_mc(get_aln):
    mc = get_no_init_model_coll(get_aln)
    return mc


@pytest.fixture()
def init_mc(get_aln):
    mc = get_init_model_coll(get_aln)
    return mc


def test_model_results_construction(no_init_mc, init_mc):
    assert isinstance(no_init_mc, model_collection_result)
    assert isinstance(init_mc, model_collection_result)


def test_no_fg(get_aln_no_fg):
    get_no_init_model_coll(get_aln_no_fg)
    get_init_model_coll(get_aln_no_fg)
    get_aln_no_fg.info = None
    get_no_init_model_coll(get_aln_no_fg)
    get_init_model_coll(get_aln_no_fg)


def test_get_no_init_hypothesis_app_run(tmp_path, dstore_instance):
    reader = io.load_db()
    outpath = tmp_path / "tempdir.tinydb"
    writer = io.write_db(outpath)
    process = reader + get_no_init_hypothesis + writer

    process.apply_to(dstore_instance[:1])

    assert len(process.data_store.summary_incomplete) == 0


def test_get_init_hypothesis_app_run(tmp_path, dstore_instance):
    reader = io.load_db()
    outpath = tmp_path / "tempdir.tinydb"
    writer = io.write_db(outpath)
    process = reader + get_init_hypothesis + writer

    process.apply_to(dstore_instance[:1])

    assert len(process.data_store.summary_incomplete) == 0


def test_get_lrt(mcr_dstore):
    loader = io.load_db()
    gn_mc = loader(mcr_dstore[0])

    hyp = get_lrt(gn_mc)
    print(hyp)
