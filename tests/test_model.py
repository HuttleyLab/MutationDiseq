from cogent3 import get_model

from mdeq import model


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
