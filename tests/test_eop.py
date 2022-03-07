import pathlib

import numpy
import pytest

from cogent3 import make_aligned_seqs
from cogent3.app import io

from mdeq.eop import adjacent_eop, edge_EOP


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def dstore_instance():
    return io.get_data_store(DATADIR / "3000bp.tinydb")


@pytest.fixture()
def multiple_alns():
    _seqs1 = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    _seqs2 = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCGTTAATGCTTGAAACCAGCAGTTATTTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACTGCTTGAGAACAGTTTGTTACTCACTATT",
    }

    _seqs3 = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCGTTAATGCTTGATGCCAGCAGTAAATTGTCCAATACT",
        "Rhesus": "GACTAGTTATTACTGCATGAGAACAGTATGTTACTCACTATT",
    }
    aln1 = make_aligned_seqs(_seqs1, moltype="dna")
    aln2 = make_aligned_seqs(_seqs2, moltype="dna")
    aln3 = make_aligned_seqs(_seqs3, moltype="dna")

    return [aln1, aln2, aln3]


@pytest.fixture()
def diff_length_alns():
    _seqs1 = {
        "Human": "GCCAGCTCATTACAGCATGAGAACAGCAGTTTATTACTCACT",
        "Bandicoot": "GACTCATTAATGCTTGAAACCAGCAGTTTATTGTCCAACACT",
        "Rhesus": "GCCAGCTCATTACAGCATGAGAACAGTTTGTTACTCACTATT",
    }

    _seqs2 = {
        "Human": "GCCAGCTCATTACAGCAT",
        "Bandicoot": "GACTCATTATTGCTTGAG",
        "Rhesus": "GCCAGCTCATTACAGCGT",
    }

    aln1 = make_aligned_seqs(_seqs1, moltype="dna")
    aln2 = make_aligned_seqs(_seqs2, moltype="dna")
    return [aln1, aln2]


loader = io.load_db()


def test_edge_EOP_construction(dstore_instance):

    aln = loader(dstore_instance[0])
    names = ["758", "443154"]

    eop = edge_EOP(aln, names)

    assert isinstance(eop.LR, float)
    assert 0 <= eop.get_LRT_stats().to_dict(flatten=True)[(0, "p")] <= 1


def test_adjacent_eop_same_aln(dstore_instance):

    aln1 = loader(dstore_instance[1])
    aln2 = loader(dstore_instance[1])

    lr = adjacent_eop([aln1, aln2], "758").LR
    numpy.testing.assert_almost_equal(lr, 0, decimal=5)

# todo check whether the multi-locus LF object can simulate alignments

