import pathlib

import pytest

from click.testing import CliRunner
from cogent3.app import io

from mdeq import (
    aeop,
    convergence,
    db_summary,
    extract_pvalues,
    get_obj_type,
    make_adjacent,
    make_controls,
    prep,
    slide,
    sql_loader,
    teop,
    toe,
)
from mdeq._click_options import (
    _gene_order_table,
    _valid_sqlitedb_input,
    _valid_sqlitedb_output,
)
from mdeq.utils import CompressedValue


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("sqlitedb")


@pytest.fixture(autouse=True)
def workingdir(tmp_dir, monkeypatch):
    # this set's the working directory for all tests in this module
    # as a tmp dir
    monkeypatch.chdir(tmp_dir)


def test_validation_input_path():
    valid = DATADIR / "300bp.sqlitedb"
    assert _valid_sqlitedb_input(*[valid])
    invalid_suffix = DATADIR / "brca1.fasta"
    with pytest.raises(ValueError):
        _valid_sqlitedb_input(*[invalid_suffix])

    invalid_path = DATADIR / "300bp.t"
    with pytest.raises(ValueError):
        _valid_sqlitedb_input(*[invalid_path])


def test_validation_output_path():
    valid = "300bp.sqlitedb"
    assert _valid_sqlitedb_output(*[valid])
    invalid_suffix = DATADIR / "brca1.fasta"
    with pytest.raises(ValueError):
        _valid_sqlitedb_output(*[invalid_suffix])


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
    types = {
        "300bp.sqlitedb": "ArrayAlignment",
        "toe-300bp.sqlitedb": "compact_bootstrap_result",
    }
    for path, expect in types.items():
        dstore = io.get_data_store(DATADIR / path)
        ty = get_obj_type(dstore)
        assert ty == expect


@pytest.fixture(scope="session")
def fasta(tmp_dir):
    """write a few fasta formatted flat files"""
    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = pathlib.Path(tmp_dir / "fasta")
    outpath.mkdir(exist_ok=True)
    dstore = io.get_data_store(inpath, limit=5)
    # make different lengths
    lengths = [250, 300, 600, 700, 1000]
    for length, m in zip(lengths, dstore):
        aln = loader(m)[:length]
        fp = outpath / f"{m.name.replace('json', 'fasta')}"
        aln.write(fp)
    return outpath


def test_prep_invalid_input(runner, tmp_dir, fasta):
    """fail if a directory and a db are provided or directory and no suffix"""
    from cogent3.app import io

    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = tmp_dir / "output.sqlitedb"
    args = f"-id {fasta} -i {inpath} -o {outpath}".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output

    args = f"-id {fasta} -o {outpath}".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output


def test_prep_defaults(runner, tmp_dir, fasta):
    """remove unamgious characters, exclude alignments < 300"""
    from cogent3.app import io

    outpath = tmp_dir / "output.sqlitedb"
    args = f"-id {fasta} -su fasta -o {outpath} -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)

    dstore = io.get_data_store(outpath)
    assert len(dstore) == 4
    assert len(dstore.incomplete) == 1
    assert all(len(loader(m)) >= 300 for m in dstore)


def test_prep_min_length(runner, tmp_dir, fasta):
    """min_length arg works"""
    from cogent3.app import io

    outpath = tmp_dir / "output.sqlitedb"
    args = f"-id {fasta} -su fasta -o {outpath} --min_length 600 -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = io.get_data_store(outpath)
    assert len(dstore) == 3
    assert len(dstore.incomplete) == 2


def test_prep_codon_pos(runner, tmp_dir, fasta):
    """codon_pos arg works"""
    from cogent3.app import io

    outpath = tmp_dir / "output.sqlitedb"
    args = f"-id {fasta} -su fasta -o {outpath} -c 1 -ml {600//3} -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = io.get_data_store(outpath)
    assert len(dstore) == 3  # since alignment length is reduced by 1/3
    assert len(dstore.incomplete) == 2


def test_prep_fg_edge(runner, tmp_dir, fasta):
    """fg_edge arg works"""
    from cogent3.app import io

    outpath = tmp_dir / "output.sqlitedb"
    # fail when name missing
    args = f"-id {fasta} -su fasta -o {outpath} --fg_edge abcde -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output

    # succeed
    fg_edge = "73021"
    args = f"-id {fasta} -su fasta -o {outpath} --fg_edge {fg_edge} -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    dstore = io.get_data_store(outpath)
    for m in dstore:
        aln = loader(m)
        assert aln.info.fg_edge == fg_edge


def test_toe_exercise(runner, tmp_dir):
    inpath = DATADIR / "300bp.sqlitedb"
    outpath = tmp_dir / "toe.sqlitedb"
    r = runner.invoke(
        toe, [f"-i{inpath}", f"-o{outpath}", "-t", "-n4", "-O"], catch_exceptions=False
    )
    assert r.exit_code == 0, r.output
    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.sqlitedb"
    r = runner.invoke(
        toe,
        ["-i", f"{invalidinput}", "-o", f"{outpath}", "-t", "-n4", "-O"],
        catch_exceptions=False,
    )
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output
    # load the result and check values
    dstore = io.get_data_store(outpath)
    read = sql_loader()
    result = read(dstore[0])
    assert isinstance(result.observed["data"], bytes)
    assert isinstance(
        CompressedValue(result.observed["data"]).decompressed.decode("utf8"), str
    )


loader = sql_loader()


def test_convergence(runner, tmp_dir):
    from cogent3.app import io

    from mdeq.convergence import delta_nabla

    inpath = DATADIR / "toe-300bp.sqlitedb"
    outpath = tmp_dir / "delme.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-O"]
    # convergence(args)
    r = runner.invoke(convergence, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    # now load the saved records and check they're delta_nabla instances
    dstore = io.get_data_store(outpath)
    results = [loader(m) for m in dstore]
    assert {type(r) for r in results} == {delta_nabla}
    assert len(dstore) == len(results)

    # now with incorrect input
    invalidinput = DATADIR / "300bp.sqlitedb"
    args[0] = f"-i{invalidinput}"
    r = runner.invoke(convergence, args, catch_exceptions=False)
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output


@pytest.fixture(scope="session")
def adjacent_path(runner, tmp_dir):
    from cogent3.app import io

    from mdeq.adjacent import grouped

    inpath = DATADIR / "apes-align.sqlitedb"
    gene_order = DATADIR / "gene_order.tsv"
    outpath = tmp_dir / "adjacent.sqlitedb"
    r = runner.invoke(
        make_adjacent,
        [f"-i{inpath}", f"-g{gene_order}", f"-o{outpath}", "-t", "-O"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output

    dstore = io.get_data_store(outpath)
    results = [loader(m) for m in dstore]
    for r in results:
        assert isinstance(r, grouped)
        assert len(r.elements) == 2

    return outpath


def test_aeop_exercise(runner, tmp_dir, adjacent_path):
    # We're using the result created in adjacent_path as input here
    inpath = adjacent_path
    outpath = tmp_dir / "aeop.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-t", "-O"]
    r = runner.invoke(aeop, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.sqlitedb"
    args[0] = f"-i{invalidinput}"
    r = runner.invoke(aeop, args, catch_exceptions=False)
    assert r.exit_code != 0
    # checking the error message
    assert "not one of the expected types" in r.output


def test_aeop_exercise_shared_mprobs(runner, tmp_dir, adjacent_path):
    # We're using the result created in adjacent_path as input here
    inpath = adjacent_path
    outpath = tmp_dir / "aeop.sqlitedb"
    r = runner.invoke(
        aeop,
        [f"-i{inpath}", f"-o{outpath}", "-t", "-O", "--share_mprobs"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


def test_teop_exercise(runner, tmp_dir):
    from cogent3.app import io, result

    # We're using the result created in adjacent_path as input here
    inpath = DATADIR / "apes-align.sqlitedb"
    outpath = tmp_dir / "teop.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-e", "Human,Chimp", "-t", "-O"]
    # teop(args)
    r = runner.invoke(teop, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = io.get_data_store(outpath)
    results = [loader(m) for m in dstore]
    for r in results:
        assert isinstance(r, result.hypothesis_result)

    # now with incorrect input
    invalidinput = DATADIR / "toe-300bp.sqlitedb"
    args[0] = f"-i{invalidinput}"
    # teop(args)
    r = runner.invoke(teop, args, catch_exceptions=False)
    assert r.exit_code != 0
    assert "not one of the expected types" in r.output


def exercise_make_controls(runner, inpath, tmp_dir, analysis, result_type):
    from cogent3.app import io

    controls = (
        "neg_control",
        "pos_control",
    )
    for ctl in controls:
        outpath = tmp_dir / pathlib.Path(f"{analysis}-{ctl}-{inpath.stem}.sqlitedb")
        control = "-ve" if "neg" in ctl else "+ve"
        for seed in (None, 123):
            args = [
                "-i",
                f"{inpath}",
                "-a",
                analysis,
                "--controls",
                control,
                "-od",
                f"{tmp_dir}",
                "-O",
                f"-s{seed}",
            ]
            args = args if seed else args[:-1]
            # make_controls(args)  # useful for debugging
            r = runner.invoke(make_controls, args, catch_exceptions=False)
            assert r.exit_code == 0, r.output

            dstore = io.get_data_store(outpath)
            results = [loader(m) for m in dstore]
            for r in results:
                assert isinstance(r, result_type)

            dstore.close()

    # now with incorrect input
    invalidinput = DATADIR / "300bp.sqlitedb"
    args[1] = f"{invalidinput}"
    r = runner.invoke(make_controls, args, catch_exceptions=False)
    assert r.exit_code != 0
    # checking the error message
    assert "does not match expected" in r.output


def test_make_controls_aeop_exercise(runner, tmp_dir):
    from mdeq.adjacent import grouped

    inpath = DATADIR / "aeop-apes.sqlitedb"
    exercise_make_controls(runner, inpath, tmp_dir, "aeop", grouped)


def test_make_controls_teop_exercise(runner, tmp_dir):
    from cogent3 import ArrayAlignment

    inpath = DATADIR / "teop-apes.sqlitedb"
    exercise_make_controls(runner, inpath, tmp_dir, "teop", ArrayAlignment)


def test_make_controls_toe_exercise(runner, tmp_dir):
    from cogent3 import ArrayAlignment

    inpath = DATADIR / "toe-300bp.sqlitedb"
    exercise_make_controls(runner, inpath, tmp_dir, "toe", ArrayAlignment)


def test_sqlitedb_summary(runner):
    inpath = DATADIR / "toe-300bp.sqlitedb"
    r = runner.invoke(db_summary, ["-i", inpath], catch_exceptions=False)
    assert r.exit_code == 0, r.output


def test_extract_pvalues(runner, tmp_dir):
    args = ["-id", str(DATADIR), "-g", "toe*", "-od", str(tmp_dir)]
    r = runner.invoke(extract_pvalues, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    # should not fail if I just give it the DATADIR due to data type mismatch since it will ignore dstores with wrong data type
    args = ["-id", str(DATADIR), "-od", str(tmp_dir)]
    r = runner.invoke(extract_pvalues, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output


def test_slide(runner, tmp_dir):
    """exercise slide"""
    # produces expected number of sliced alignments with given input
    # aligns are 3k long, window 700, setting step to 500 give 5 sub-alignments, times 5
    # alignments gives 25
    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = tmp_dir / "output.sqlitedb"
    window_size = 600
    step = 500
    min_length = 200
    args = f"-i {inpath} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    dstore = io.get_data_store(outpath)
    assert len(dstore) == 25, dstore
    assert len(dstore.incomplete) == 0, dstore.incomplete
    assert len(dstore.logs) == 1, dstore.logs
    dstore.close()

    # fails when invalid input data type
    args = f"-i {DATADIR / 'teop-apes.sqlitedb'} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 1, r.output

    # setting minlength greater than window size causes exit
    min_length = window_size + 1
    args = f"-i {inpath} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 1, r.output
