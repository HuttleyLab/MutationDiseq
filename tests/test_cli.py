import pathlib

import pytest
from click.testing import CliRunner
from cogent3 import load_table, open_data_store

from mdeq import (
    aeop,
    convergence,
    db_summary,
    extract_nabla_c,
    extract_pvalues,
    load_from_sqldb,
    make_adjacent,
    make_controls,
    prep,
    slide,
    teop,
    toe,
)
from mdeq._click_options import (
    _gene_order_table,
    _valid_sqlitedb_input,
    _valid_sqlitedb_output,
)
from mdeq.utils import CompressedValue, matches_type

__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]


DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return pathlib.Path(tmpdir_factory.mktemp("sqlitedb"))


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
        "300bp.sqlitedb": "Alignment",
        "toe-300bp.sqlitedb": "compact_bootstrap_result",
    }
    for path, expect in types.items():
        dstore = open_data_store(DATADIR / path)
        assert matches_type(dstore, (expect,))


@pytest.fixture(scope="session")
def fasta(tmp_dir):
    """write a few fasta formatted flat files"""
    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = pathlib.Path(tmp_dir / "fasta")
    outpath.mkdir(exist_ok=True)
    dstore = open_data_store(inpath, limit=5)
    # make different lengths
    lengths = [250, 300, 600, 700, 1000]
    for length, m in zip(lengths, dstore, strict=False):
        aln = loader(m)[:length]
        fp = outpath / f"{m.unique_id.replace('json', 'fasta')}"
        aln.write(fp)
    return outpath


def test_prep_invalid_input(runner, tmp_dir, fasta):
    """fail if a directory and a db are provided or directory and no suffix"""
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
    outpath = tmp_dir / "output.sqlitedb"
    args = f"-id {fasta} -su fasta -o {outpath} -O".split()
    runner.invoke(prep, args, catch_exceptions=False)

    dstore = open_data_store(outpath)
    assert len(dstore.completed) == 4
    assert len(dstore.not_completed) == 1
    assert all(len(loader(m)) >= 300 for m in dstore.completed)


def test_prep_min_length(runner, tmp_dir, fasta):
    """min_length arg works"""
    outpath = tmp_dir / "output.sqlitedb"
    outpath.unlink(missing_ok=True)
    args = f"-id {fasta} -su fasta -o {outpath} --min_length 600 -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = open_data_store(outpath)
    assert len(dstore.completed) == 3
    assert len(dstore.not_completed) == 2


def test_prep_codon_pos(runner, tmp_dir, fasta):
    """codon_pos arg works"""
    outpath = tmp_dir / "output.sqlitedb"
    outpath.unlink(missing_ok=True)
    args = f"-id {fasta} -su fasta -o {outpath} -c 1 -ml {600 // 3} -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = open_data_store(outpath)
    assert len(dstore.completed) == 3  # since alignment length is reduced by 1/3
    assert len(dstore.not_completed) == 2


def test_prep_fg_edge(runner, tmp_dir, fasta):
    """fg_edge arg works"""
    outpath = tmp_dir / "output.sqlitedb"
    outpath.unlink(missing_ok=True)
    # fail when name missing
    args = f"-id {fasta} -su fasta -o {outpath} --fg_edge abcde -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code != 0, r.output

    # succeed
    fg_edge = "73021"
    args = f"-id {fasta} -su fasta -o {outpath} --fg_edge {fg_edge} -O".split()
    r = runner.invoke(prep, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    dstore = open_data_store(outpath)
    for m in dstore.completed:
        aln = loader(m)
        assert aln.info.fg_edge == fg_edge


def test_toe_exercise(runner, tmp_dir):
    inpath = DATADIR / "300bp.sqlitedb"
    outpath = tmp_dir / "toe.sqlitedb"
    r = runner.invoke(
        toe,
        [f"-i{inpath}", f"-o{outpath}", "-t", "-n4", "-O"],
        catch_exceptions=False,
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
    dstore = open_data_store(outpath)
    result = loader(dstore[0])
    assert isinstance(result.observed["data"], bytes)
    assert isinstance(CompressedValue(result.observed["data"]).as_primitive, dict)


loader = load_from_sqldb()


def test_convergence(runner, tmp_dir):
    from mdeq.convergence import nabla_c

    inpath = DATADIR / "toe-300bp.sqlitedb"
    outpath = tmp_dir / "delme.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-O"]
    r = runner.invoke(convergence, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    # now load the saved records and check they're nabla_c instances
    dstore = open_data_store(outpath)

    results = [loader(m) for m in dstore]
    assert {type(r) for r in results} == {nabla_c}
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

    dstore = open_data_store(outpath)
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
    outpath.unlink(missing_ok=True)
    r = runner.invoke(
        aeop,
        [f"-i{inpath}", f"-o{outpath}", "-t", "-O", "--share_mprobs"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output


def test_teop_exercise(runner, tmp_dir):
    from cogent3.app import result

    # We're using the result created in adjacent_path as input here
    inpath = DATADIR / "apes-align.sqlitedb"
    outpath = tmp_dir / "teop.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-e", "Human,Chimp", "-t", "-O"]
    # teop(args)
    r = runner.invoke(teop, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    dstore = open_data_store(outpath)
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
    controls = (
        "neg_control",
        "pos_control",
    )
    for ctl in controls:
        outpath = tmp_dir / pathlib.Path(f"{analysis}-{ctl}-{inpath.stem}.sqlitedb")
        control = "-ve" if "neg" in ctl else "+ve"
        for seed in (None, 123):
            outpath.unlink(missing_ok=True)
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

            dstore = open_data_store(outpath)
            results = [loader(m) for m in dstore]
            for r in results:
                assert isinstance(r, result_type)

            assert len(dstore.completed) > 0
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
    from cogent3.core.new_alignment import Alignment

    inpath = DATADIR / "teop-apes.sqlitedb"
    exercise_make_controls(runner, inpath, tmp_dir, "teop", Alignment)


def test_make_controls_toe_exercise(runner, tmp_dir):
    from cogent3.core.new_alignment import Alignment

    inpath = DATADIR / "toe-300bp.sqlitedb"
    exercise_make_controls(runner, inpath, tmp_dir, "toe", Alignment)


def test_sqlitedb_summary(runner):
    inpath = DATADIR / "toe-300bp.sqlitedb"
    r = runner.invoke(db_summary, ["-i", inpath], catch_exceptions=False)
    assert r.exit_code == 0, r.output


def test_extract_pvalues(runner, tmp_dir):
    args = ["-id", str(DATADIR), "-od", str(tmp_dir), "-g", "toe*.sqlitedb"]
    r = runner.invoke(extract_pvalues, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output


def test_extract_nabla_c(runner, tmp_dir):
    # generate the convergence results
    inpath = DATADIR / "toe-300bp.sqlitedb"
    outpath = tmp_dir / "delme.sqlitedb"
    args = [f"-i{inpath}", f"-o{outpath}", "-O"]
    r = runner.invoke(convergence, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    # run extract command on output of convergence command
    tsv_file = outpath.with_suffix(".tsv")
    args = f"-i {outpath!s} -od {tsv_file.parent}".split()
    r = runner.invoke(extract_nabla_c, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    # table num rows should equal number of completeds in convergence dstore
    dstore = open_data_store(outpath)
    t = load_table(tsv_file)
    assert t.shape[0] == len(dstore.completed)


def test_slide(runner, tmp_dir):
    """exercise slide"""
    # produces expected number of sliced alignments with given input
    # aligns are 3k long, window 700, setting step to 500 give 5 sub-alignments, times 5
    # alignments gives 25
    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = tmp_dir / "output.sqlitedb"
    outpath.unlink(missing_ok=True)
    window_size = 600
    step = 500
    min_length = 200
    args = f"-i {inpath} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 0, r.output
    dstore = open_data_store(outpath)
    assert len(dstore.completed) >= 25, dstore
    assert len(dstore.not_completed) >= 0, dstore.not_completed
    assert len(dstore.logs) == 1, dstore.logs
    # the info attribute should have the correct fg_edge setting
    expect = "73021"
    for m in dstore:
        aln = loader(m)
        assert aln.info["fg_edge"] == expect

    dstore.close()


def test_slide_exit(runner, tmp_dir):
    # fails when invalid input data type
    inpath = DATADIR / "3000bp.sqlitedb"
    outpath = tmp_dir / "output.sqlitedb"
    window_size = 600
    step = 500
    min_length = 200
    args = f"-i {DATADIR / 'teop-apes.sqlitedb'} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 1, r.output

    # fail if minlength > window size
    min_length = window_size + 1
    args = f"-i {inpath} -o {outpath} -wz {window_size} -st {step} -ml {min_length} -O".split()
    r = runner.invoke(slide, args, catch_exceptions=False)
    assert r.exit_code == 1, r.output
