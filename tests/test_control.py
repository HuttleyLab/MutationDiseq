import pathlib

import pytest

from cogent3.app import io
from cogent3.core.alignment import ArrayAlignment

from mdeq import control
from mdeq.adjacent import grouped


__author__ = "Gavin Huttley"
__credits__ = ["Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def opt_args():
    return {"max_evaluations": 100, "limit_action": "ignore", "max_restarts": 1}


@pytest.fixture()
def apes_dstore():
    return io.get_data_store(DATADIR / "apes-align.tinydb")


@pytest.fixture(scope="session")
def toe_result():
    inpath = DATADIR / "toe-300bp.tinydb"
    loader = io.load_db()

    dstore = io.get_data_store(inpath)
    return loader(dstore[0])


@pytest.fixture(scope="session")
def model_result():
    inpath = DATADIR / "toe-300bp.tinydb"
    loader = io.load_db()

    dstore = io.get_data_store(inpath)
    return loader(dstore[0]).observed["GSN"]


def test_select_model_result(toe_result):
    def validate_selected(result, name):
        selector = control.select_model_result(name)
        got = selector(result)
        assert got.lf._model.name == name

    validate_selected(toe_result, "GSN")
    validate_selected(toe_result, "GN")


def test_select_model_result_model(model_result):
    selector = control.select_model_result(None)
    got = selector(model_result)
    assert got is model_result


def test_select_aeop(apes_dstore, opt_args):
    from cogent3 import ArrayAlignment

    from mdeq.eop import ALT_AEOP, NULL_AEOP, adjacent_eop

    def get_selected(result, name):
        selector = control.select_model_result(name)
        return selector(result)

    loader = io.load_db()
    alns = [loader(apes_dstore[i]) for i in (2, 4)]
    for n, a in zip(("a", "b"), alns):
        a.info.name = n
    grp = grouped(("a", "b"))
    grp.elements = alns

    opt_args["max_evaluations"] = 10
    app = adjacent_eop(opt_args=opt_args)
    result = app(grp)
    n = get_selected(result, NULL_AEOP)
    # should have multiple loci in lf itself
    a = n.lf.get_param_value("alignment", locus="a")

    assert isinstance(a, ArrayAlignment)
    a = get_selected(result, ALT_AEOP)
    assert len(a) == 2
    # keyed by alignments
    assert set(a) == {"a", "b"}


def test_select_teop(apes_dstore, opt_args):
    from cogent3.app.result import model_result

    from mdeq.eop import ALT_TEOP, NULL_TEOP, temporal_eop

    def get_selected(result, name):
        selector = control.select_model_result(name)
        return selector(result)

    loader = io.load_db()
    aln = loader(apes_dstore[0])

    opt_args["max_evaluations"] = 10
    app = temporal_eop(edge_names=["Human", "Chimp"], opt_args=opt_args)
    result = app(aln)
    n = get_selected(result, NULL_TEOP)
    a = get_selected(result, ALT_TEOP)
    for r in (n, a):
        assert isinstance(r, model_result)


def test_gen_toe_null(toe_result):
    from mdeq.lrt import NULL_TOE

    selector = control.select_model_result(NULL_TOE)
    gen = control.control_generator(selector, 2)
    got = gen(toe_result)
    assert len(got) == 2
    # successive calls work too
    got = gen(toe_result)
    assert len(got) == 2


def test_gen_toe_alt(toe_result):
    from mdeq.lrt import ALT_TOE

    selector = control.select_model_result(ALT_TOE)
    gen = control.control_generator(selector, 2)
    got = gen(toe_result)
    assert len(got) == 2


@pytest.fixture(scope="session")
def aeop_result():
    inpath = DATADIR / "aeop-apes.tinydb"

    dstore = io.get_data_store(inpath)
    loader = io.load_db()
    return [loader(m) for m in dstore]


@pytest.fixture(scope="session")
def teop_result():
    inpath = DATADIR / "teop-apes.tinydb"

    dstore = io.get_data_store(inpath)
    loader = io.load_db()
    return [loader(m) for m in dstore]


def test_gen_aeop_null(aeop_result):
    from mdeq.eop import NULL_AEOP

    def validate_for_result(generator, result, num_reps):
        got = generator(result)
        assert len(got) == num_reps
        for record in got:
            assert isinstance(record, grouped)
            assert all(isinstance(e, ArrayAlignment) for e in record.elements)

    selector = control.select_model_result(NULL_AEOP)
    num_reps = 2
    gen = control.control_generator(selector, num_reps=num_reps)
    for result in aeop_result:
        validate_for_result(gen, result, num_reps)


def test_gen_aeop_alt(aeop_result):
    from mdeq.eop import ALT_AEOP

    def validate_for_result(generator, result, num_reps):
        got = generator(result)
        assert len(got) == num_reps
        for record in got:
            assert isinstance(record, grouped)
            assert all(isinstance(e, ArrayAlignment) for e in record.elements)

    selector = control.select_model_result(ALT_AEOP)
    num_reps = 2
    gen = control.control_generator(selector, num_reps=num_reps)
    for result in aeop_result:
        validate_for_result(gen, result, num_reps)


def test_gen_teop_null(teop_result):
    from mdeq.eop import NULL_TEOP

    def validate_for_result(generator, result, num_reps):
        got = generator(result)
        assert len(got) == num_reps
        for r in got:
            assert isinstance(r, ArrayAlignment)

    selector = control.select_model_result(NULL_TEOP)
    num_reps = 2
    gen = control.control_generator(selector, num_reps=num_reps)
    for result in teop_result:
        validate_for_result(gen, result, num_reps)


def test_gen_teop_alt(teop_result):
    from mdeq.eop import ALT_TEOP

    def validate_for_result(generator, result, num_reps):
        got = generator(result)
        assert len(got) == num_reps
        for r in got:
            assert isinstance(r, ArrayAlignment)

    selector = control.select_model_result(ALT_TEOP)
    num_reps = 2
    gen = control.control_generator(selector, num_reps=num_reps)
    for result in teop_result:
        validate_for_result(gen, result, num_reps)
