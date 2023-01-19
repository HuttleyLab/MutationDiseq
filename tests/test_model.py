import pathlib

from cogent3 import get_model, open_data_store

from mdeq import model
from mdeq.sqlite_data_store import load_from_sql


__author__ = "Katherine Caley"
__credits__ = ["Katherine Caley", "Gavin Huttley"]

DATADIR = pathlib.Path(__file__).parent / "data"

loader = load_from_sql()


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
    from cogent3.app import composable

    path = DATADIR / "toe-300bp-new.sqlitedb"
    dstore = open_data_store(path, limit=1)
    r = loader(dstore[0])
    r.deserialised_values()
    models = []
    for mod in ("GSN", "GN"):
        models.append(r.observed[mod])

    app = model.mles_within_bounds()
    for mod in models:
        got = app(mod)
        assert got is mod

    # set lower value of 1.0 (these models have min val of 1), should return NotCompleted
    app = model.mles_within_bounds(lower=1.0)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # set upper value of 10 (both models have upper vals >20), should return NotCompleted
    app = model.mles_within_bounds(upper=10)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # set both lower and upper, should return NotCompleted
    app = model.mles_within_bounds(lower=1, upper=10)
    for mod in models:
        got = app(mod)
        assert isinstance(got, composable.NotCompleted)

    # value of length should trigger NotCompleted
    mod.lf.set_param_rule("length", init=1e-6)
    app = model.mles_within_bounds(lower=0.5)
    got = app(mod)
    assert got is mod
