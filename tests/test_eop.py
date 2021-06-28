import numpy
import pytest
from cogent3 import make_aligned_seqs
from cogent3.app import io
from kath_analysis.eop import EOP


@pytest.fixture()
def dstore_instance():
    dstore = io.get_data_store(
        "~/repos/data/microbial/synthetic/758_443154_73021/3000bp/alns.tinydb"
    )
    return dstore


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


def test_construction(dstore_instance):

    aln1 = loader(dstore_instance[0])
    aln2 = loader(dstore_instance[1])

    eop = EOP([aln1, aln2])

    assert isinstance(eop.LR, float)
    assert 0 <= eop.get_LRT_stats().to_dict(flatten=True)[(0, "p")] <= 1


def test_eop_same_aln(dstore_instance):

    aln1 = loader(dstore_instance[1])
    aln2 = loader(dstore_instance[1])

    lr = EOP([aln1, aln2]).LR
    numpy.testing.assert_almost_equal(lr, 0, decimal=5)


def test_get_rel_entropies_same_aln(dstore_instance):

    loader = io.load_db()

    aln1 = loader(dstore_instance[1])
    aln2 = loader(dstore_instance[1])

    eop = EOP([aln1, aln2])
    re = list(eop.get_relative_entropies().values())

    numpy.testing.assert_almost_equal(re[0], re[1], decimal=4)


def test_eop_three_alns(multiple_alns):
    eop = EOP(multiple_alns)
    assert isinstance(eop.LR, float)
    assert 0 <= eop.get_LRT_stats().to_dict(flatten=True)[(0, "p")] <= 1
    eop.get_relative_entropies()


def test_eop_different_lengths(diff_length_alns):
    eop = EOP(diff_length_alns)
    assert isinstance(eop.LR, float)
    assert 0 <= eop.get_LRT_stats().to_dict(flatten=True)[(0, "p")] <= 1
    eop.get_relative_entropies()


def test_eop_short_seq():
    dstore = io.get_data_store(
        "/Users/katherine/repos/results/aim_2/microbial/758_443154_73021/300bp_10000_repeats.tinydb"
    )
    loader = io.load_db()
    aln1 = loader(dstore[9])
    aln2 = loader(dstore[10])

    eop = EOP([aln1, aln2])
