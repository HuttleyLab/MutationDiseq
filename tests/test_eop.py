import pathlib

import numpy
import pytest

from cogent3 import make_aligned_seqs
from cogent3.app import io
from cogent3.app.result import hypothesis_result

from mdeq.eop import adjacent_eop, edge_EOP


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tinydb")


@pytest.fixture(scope="session")
def opt_args():
    return dict(max_restarts=1, max_evaluations=50, limit_action="ignore")


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


def test_adjacent_eop_same_aln(dstore_instance, tmp_dir, opt_args):
    from cogent3.util.dict_array import DictArray

    from mdeq.adjacent import grouped

    aln = loader(dstore_instance[4])

    aln.info.pop("fg_edge")
    grp = grouped(("a", "b"))
    grp.elements = [aln, aln]

    opt_args["max_evaluations"] = 100
    app = adjacent_eop(opt_args=opt_args)
    result = app(grp)
    numpy.testing.assert_almost_equal(result.LR, 0, decimal=5)
    # just one mrpobs
    mprobs = result.null.lf.get_motif_probs()
    assert isinstance(mprobs, DictArray)

    # allow mprobs to be different
    app = adjacent_eop(opt_args=opt_args, share_mprobs=False)
    result = app(grp)
    mprobs = result.null.lf.get_motif_probs()
    assert isinstance(mprobs, dict)
    assert len(mprobs) == 2


# todo check whether the multi-locus LF object can simulate alignments
# Answer - it can, e.g. lf.simulate_alignment(locus="a")


def test_adjacent_eop(multiple_alns, opt_args):
    from mdeq.adjacent import grouped

    identifiers = []
    for i, aln in enumerate(multiple_alns):
        identifiers.append(f"name-{i}")
        aln.info.name = identifiers[-1]

    grp = grouped(identifiers=identifiers)
    grp.elements = multiple_alns[:2]
    test_adjacent = adjacent_eop(opt_args=opt_args)
    # works if no fg edge specified
    got = test_adjacent(grp)
    assert isinstance(got, hypothesis_result)
    for aln in multiple_alns:
        aln.info.fg_edge = "Human"
    # although elements is immutable, the alignment instance members are
    # not so the data is actually differemt
    got = test_adjacent(grp)
    assert isinstance(got, hypothesis_result)
