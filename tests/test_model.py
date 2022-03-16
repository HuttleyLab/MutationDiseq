import pathlib

from cogent3 import get_model

from mdeq import model


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"


def test_get_gsn():
    """model registry process works and we can get a GSN instance."""
    from cogent3.evolve.ns_substitution_model import GeneralStationary

    sm = get_model("GSN")
    assert isinstance(sm, GeneralStationary)

    # then directly
    sm = model.GSN()
    assert isinstance(sm, GeneralStationary)


def test_make_gsn_app():
    """correctly makes a GSN model app."""
    from cogent3.evolve.ns_substitution_model import GeneralStationary

    app = model.GS_sm()
    assert isinstance(app._sm, GeneralStationary)


def test_mles_at_bounds():
    from cogent3.app import composable, io

    loader = io.load_db()
    path = DATADIR / "toe-300bp.tinydb"
    dstore = io.get_data_store(path, limit=1)
    models = [loader(dstore[0]).observed[mod] for mod in ("GSN", "GN")]
    app = model.mles_near_bounds()
    for mod in models:
        got = app(mod)
        assert got is mod

    # set lower value of 1.0 (these models have min val of 1), should return NotCompleted
    app = model.mles_near_bounds(lower=1.0)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # set upper value of 10 (both models have upper vals >20), should return NotCompleted
    app = model.mles_near_bounds(upper=10)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # set both lower and upper, should return NotCompleted
    app = model.mles_near_bounds(lower=1, upper=10)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # value of length should trigger NotCompleted
    mod.lf.set_param_rule("length", init=1e-6)
    app = model.mles_near_bounds(lower=0.5)
    got = app(mod)
    assert got is mod
