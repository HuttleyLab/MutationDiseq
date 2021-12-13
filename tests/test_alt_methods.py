import pytest
from cogent3.app import io
from kath_analysis.alt_methods.squartini_arndt import (
    chi_squared_test,
    stationarity_indices,
)


@pytest.fixture()
def mcr_dstore():
    dstore = io.get_data_store(
        "/Users/katherine/repos/results/aim_2/synthetic/758_443154_73021/300bp/mcr-init.tinydb"
    )

    return dstore


@pytest.fixture()
def model_result(mcr_dstore):
    loader = io.load_db()
    result = loader(mcr_dstore[0])["mcr"]["GN"]

    return result


@pytest.fixture()
def model_result_GTR(mcr_dstore):
    loader = io.load_db()
    result = loader(mcr_dstore[0])["mcr"]["GTR"]
    return result


def test_chi_squared_test(model_result):
    result = chi_squared_test(model_result)
    print(result)


def test_stationarity_indices(model_result):
    result = stationarity_indices(model_result)

    print(result)


def test_chi_squared_test_GTR(model_result_GTR):

    result = chi_squared_test(model_result_GTR)
    assert result.to_dict(flatten=True)[(0, "p")] > 0.05


def test_stationarity_indices_GTR(model_result_GTR):
    result = stationarity_indices(model_result_GTR)

    print(result)
