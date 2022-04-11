import pathlib

import pytest

from cogent3 import make_aligned_seqs
from cogent3.app import io
from cogent3.app.evo import model_collection_result

from mdeq.sqlite_data_store import sql_loader, sql_writer
from mdeq.toe import (
    get_init_hypothesis,
    get_init_model_coll,
    get_lrt,
    get_no_init_hypothesis,
    get_no_init_model_coll,
)


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def opt_args():
    return {"max_evaluations": 1000, "limit_action": "ignore"}


@pytest.fixture()
def dstore_instance():
    return io.get_data_store(DATADIR / "3000bp.sqlitedb")


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
    return make_aligned_seqs(data=_data, moltype="dna")


@pytest.fixture()
def no_init_mc(get_aln, opt_args):
    return get_no_init_model_coll(get_aln, just_continuous=False, opt_args=opt_args)


@pytest.fixture()
def init_mc(get_aln, opt_args):
    return get_init_model_coll(get_aln, just_continuous=False, opt_args=opt_args)


def test_model_results_construction(no_init_mc, init_mc):
    assert isinstance(no_init_mc, model_collection_result)
    assert isinstance(init_mc, model_collection_result)


def test_no_fg(get_aln_no_fg, opt_args):
    get_no_init_model_coll(get_aln_no_fg, just_continuous=True, opt_args=opt_args)
    get_init_model_coll(get_aln_no_fg, just_continuous=True, opt_args=opt_args)
    get_aln_no_fg.info = None
    get_no_init_model_coll(get_aln_no_fg, just_continuous=True, opt_args=opt_args)
    get_init_model_coll(get_aln_no_fg, just_continuous=True, opt_args=opt_args)


reader = sql_loader()


def test_get_no_init_hypothesis_app_run(tmp_path, dstore_instance, opt_args):
    outpath = tmp_path / "tempdir.sqlitedb"
    writer = sql_writer(outpath)
    reader.disconnect()
    process = (
        reader
        + get_no_init_hypothesis(just_continuous=False, opt_args=opt_args)
        + writer
    )
    process.apply_to(dstore_instance[:1])
    dstore = io.get_data_store(outpath)
    assert len(dstore.summary_incomplete) == 0


def test_get_init_hypothesis_app_run(tmp_path, dstore_instance, opt_args):
    outpath = tmp_path / "tempdir.sqlitedb"
    writer = sql_writer(outpath)
    reader.disconnect()
    process = (
        reader + get_init_hypothesis(just_continuous=False, opt_args=opt_args) + writer
    )
    process.apply_to(dstore_instance[:1])
    dstore = io.get_data_store(outpath)
    assert len(dstore.summary_incomplete) == 0
