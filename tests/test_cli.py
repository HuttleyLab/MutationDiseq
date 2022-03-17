import pathlib

import pytest

from click.testing import CliRunner

from mdeq import (
    _gene_order_table,
    _valid_tinydb_input,
    _valid_tinydb_output,
    aeop,
    convergence,
    get_obj_type,
    make_adjacent,
    make_controls,
    teop,
    toe,
)


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("tinydb")


def test_validation_input_path():
    valid = DATADIR / "300bp.tinydb"
    assert _valid_tinydb_input(*[valid])
    invalid_suffix = DATADIR / "brca1.fasta"
    with pytest.raises(ValueError):
        _valid_tinydb_input(*[invalid_suffix])

    invalid_path = DATADIR / "300bp.t"
    with pytest.raises(ValueError):
        _valid_tinydb_input(*[invalid_path])


def test_validation_output_path():
    valid = "300bp.tinydb"
    assert _valid_tinydb_output(*[valid])
    invalid_suffix = DATADIR / "brca1.fasta"
    with pytest.raises(ValueError):
        _valid_tinydb_output(*[invalid_suffix])


def test_assert_valid_gene_order_table(tmp_dir):
    from cogent3 import make_table

    valid_path = tmp_dir / "valid.tsv"
    invalid_path = tmp_dir / "invalid.tsv"

    valid = make_table(data={"name": ["ENSG22"], "coord_name": [1], "start": [23]})
    valid.write(valid_path)
    assert _gene_order_table(*[str(valid_path)])

    for col in valid.header:
        invalid = valid[:, [c for c in valid.header if c != col]]
        invalid.write(invalid_path)
        with pytest.raises(ValueError):
            _gene_order_table(*[str(invalid_path)])


@pytest.fixture(scope="session")
def runner():
    """exportrc works correctly."""
    return CliRunner()


def test_get_obj_type():
    from cogent3.app.io import get_data_store

    types = {
        "300bp.tinydb": "ArrayAlignment",
        "toe-300bp.tinydb": "compact_bootstrap_result",
    }
    for path, expect in types.items():
        dstore = get_data_store(DATADIR / path)
        ty = get_obj_type(dstore)
        assert ty == expect


def test_toe_exercise(runner, tmp_dir):
    inpath = DATADIR / "300bp.tinydb"
    outpath = tmp_dir / "toe.tinydb"
    r = runner.invoke(toe, [f"-i{inpath}", f"-o{outpath}", "-t", "-n4", "-O"])
    assert r.exit_code == 0, r.output
    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.tinydb"
    r = runner.invoke(
        toe, ["-i", f"{invalidinput}", "-o", f"{outpath}", "-t", "-n4", "-O"]
    )
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output


def test_convergence(runner, tmp_dir):
    from cogent3.app import io

    from mdeq.convergence import delta_nabla

    inpath = DATADIR / "toe-300bp.tinydb"
    outpath = tmp_dir / "delme.tinydb"
    args = [f"-i{inpath}", f"-o{outpath}", "-O"]
    r = runner.invoke(convergence, args)
    assert r.exit_code == 0, r.output
    # now load the saved records and check they're delta_nabla instances
    dstore = io.get_data_store(outpath)
    loader = io.load_db()
    results = [loader(m) for m in dstore]
    assert {type(r) for r in results} == {delta_nabla}
    assert len(dstore) == len(results)

    # now with incorrect input
    invalidinput = DATADIR / "300bp.tinydb"
    args[0] = f"-i{invalidinput}"
    r = runner.invoke(convergence, args)
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output


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
    args = [f"-i{inpath}", f"-o{outpath}", "-t", "-O"]
    r = runner.invoke(aeop, args)
    assert r.exit_code == 0, r.output

    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.tinydb"
    args[0] = f"-i{invalidinput}"
    r = runner.invoke(aeop, args)
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output


def test_aeop_exercise_shared_mprobs(runner, tmp_dir, test_make_adjacent):
    # We're using the result created in test_make_adjacent as input here
    inpath = test_make_adjacent
    outpath = tmp_dir / "aeop.tinydb"
    r = runner.invoke(
        aeop, [f"-i{inpath}", f"-o{outpath}", "-t", "-O", "--share_mprobs"]
    )
    assert r.exit_code == 0, r.output


def test_teop_exercise(runner, tmp_dir):
    from cogent3.app import io, result

    # We're using the result created in test_make_adjacent as input here
    inpath = DATADIR / "apes-align.tinydb"
    outpath = tmp_dir / "teop.tinydb"
    args = [f"-i{inpath}", f"-o{outpath}", "-e'Human,Chimp'", "-t", "-O"]
    r = runner.invoke(teop, args)
    assert r.exit_code == 0, r.output

    loader = io.load_db()
    dstore = io.get_data_store(outpath)
    results = [loader(m) for m in dstore]
    for r in results:
        assert isinstance(r, result.hypothesis_result)

    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.tinydb"
    args[0] = f"-i{invalidinput}"
    r = runner.invoke(teop, args)
    assert r.exit_code != 0
    assert "not one of the expected types" in r.output


def exercise_make_controls(runner, inpath, tmp_dir, analysis, result_type):
    from cogent3.app import io

    outpaths = (
        f"{analysis}-negative_controls.tinydb",
        f"{analysis}-positive_controls.tinydb",
    )
    for path in outpaths:
        control = "-ve" if "neg" in path else "+ve"
        outpath = tmp_dir / path
        for seed in (None, 123):
            args = [
                "-i",
                f"{inpath}",
                "-a",
                analysis,
                "--controls",
                control,
                "-o",
                f"{outpath}",
                "-O",
                f"-s{seed}",
            ]
            args = args if seed else args[:-1]
            # make_controls(args)  # useful for debugging
            r = runner.invoke(make_controls, args)
            assert r.exit_code == 0, r.output

            loader = io.load_db()
            dstore = io.get_data_store(outpath)
            results = [loader(m) for m in dstore]
            for r in results:
                assert isinstance(r, result_type)

            dstore.close()

    # now with incorrect input
    invalidinput = DATADIR / "300bp.tinydb"
    args[1] = f"{invalidinput}"
    r = runner.invoke(make_controls, args)
    assert r.exit_code != 0
    # checking the error message
    assert "does not match expected" in r.output


def test_make_controls_aeop_exercise(runner, tmp_dir):
    from mdeq.adjacent import grouped

    inpath = DATADIR / "aeop-apes.tinydb"
    exercise_make_controls(runner, inpath, tmp_dir, "aeop", grouped)


def test_make_controls_teop_exercise(runner, tmp_dir):
    from cogent3 import ArrayAlignment

    inpath = DATADIR / "teop-apes.tinydb"
    exercise_make_controls(runner, inpath, tmp_dir, "teop", ArrayAlignment)


def test_make_controls_toe_exercise(runner, tmp_dir):
    from cogent3 import ArrayAlignment

    inpath = DATADIR / "toe-300bp.tinydb"
    exercise_make_controls(runner, inpath, tmp_dir, "toe", ArrayAlignment)
