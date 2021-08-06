import pytest
from cogent3.app import io

from kath_library.alt_methods.squartini_arndt import (
    chi_squared_test,
    stationarity_indices,
)


@pytest.fixture()
def mcr_dstore():
    dstore = io.get_data_store(
        f"/Users/katherine/repos/results/aim_2/synthetic/758_443154_73021/3000bp/mcr.tinydb"
    )

    return dstore


@pytest.fixture()
def model_result(mcr_dstore):
    loader = io.load_db()
    result = loader(mcr_dstore[0])["mcr"]["GN"]

    return result


def test_chi_squared_test(model_result):

    result = chi_squared_test(model_result)

    print(result)


def test_stationarity_indices(model_result):
    result = stationarity_indices(model_result)

    print(result)
