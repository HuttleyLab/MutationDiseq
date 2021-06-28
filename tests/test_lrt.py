import pytest
from cogent3 import make_aligned_seqs
from cogent3.app.evo import model_collection_result
from kath_analysis.lrt import get_init_model_coll, get_no_init_model_coll


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
