import pathlib

import pytest

from click.testing import CliRunner

from mdeq import aeop, convergence, make_adjacent, toe


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def runner():
    """exportrc works correctly"""
    return CliRunner()


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tinydb")


def test_toe_exercise(runner, tmp_dir):
    inpath = DATADIR / "300bp.tinydb"
    outpath = tmp_dir / "toe.tinydb"
    r = runner.invoke(toe, [f"-i{inpath}", f"-o{outpath}", "-t", "-n4", "-O"])
    assert r.exit_code == 0, r.output


def test_convergence(runner, tmp_dir):
    from cogent3.app import io

    from mdeq.convergence import delta_nabla

    inpath = DATADIR / "toe-300bp.tinydb"
    outpath = tmp_dir / "delme.tinydb"
    r = runner.invoke(convergence, [f"-i{inpath}", f"-o{outpath}", "-O"])
    assert r.exit_code == 0, r.output
    # now load the saved records and check they're delta_nabla instances
    dstore = io.get_data_store(outpath)
    loader = io.load_db()
    results = [loader(m) for m in dstore]
    assert {type(r) for r in results} == {delta_nabla}
    assert len(dstore) == len(results)


@pytest.fixture(scope="session")
def test_make_adjacent(runner, tmp_dir):
    from cogent3.app import io

    from mdeq.adjacent import grouped

    inpath = DATADIR / "apes-align.tinydb"
    gene_order = DATADIR / "gene_order.tsv"
    outpath = tmp_dir / "adjacent.tinydb"
    r = runner.invoke(
        make_adjacent, [f"-i{inpath}", f"-g{gene_order}", f"-o{outpath}", "-t", "-O"]
    )
    assert r.exit_code == 0, r.output

    loader = io.load_db()
    dstore = io.get_data_store(outpath)
    results = [loader(m) for m in dstore]
    for r in results:
        assert isinstance(r, grouped)
        assert len(r.elements) == 2

    return outpath


def test_aeop_exercise(runner, tmp_dir, test_make_adjacent):
    # We're using the result created in test_make_adjacent as input here
    inpath = test_make_adjacent
    outpath = tmp_dir / "aeop.tinydb"
    r = runner.invoke(aeop, [f"-i{inpath}", f"-o{outpath}", "-t", "-O"])
    assert r.exit_code == 0, r.output


def test_aeop_exercise_shared_mprobs(runner, tmp_dir, test_make_adjacent):
    # We're using the result created in test_make_adjacent as input here
    inpath = test_make_adjacent
    outpath = tmp_dir / "aeop.tinydb"
    r = runner.invoke(
        aeop, [f"-i{inpath}", f"-o{outpath}", "-t", "-O", "--share_mprobs"]
    )
    assert r.exit_code == 0, r.output
