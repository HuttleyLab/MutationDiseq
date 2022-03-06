import pathlib

import pytest

from click.testing import CliRunner

from mdeq import convergence, toe


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
