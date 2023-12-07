import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peptide_forest.regressor_model import RegressorModel

pytest._test_path = Path(__file__).parent


@pytest.fixture
def temp_model_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        yield tmp.name
    os.remove(tmp.name)


@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 10))

    y = (
        X.iloc[:, 0]
        + 2 * X.iloc[:, 1]
        - 0.5 * X.iloc[:, 2]
        + np.random.randn(100) * 0.1
    )

    return X, pd.Series(y)


@pytest.fixture
def mock_regressor_model(sample_data):
    model = RegressorModel(
        model_type="xgboost",
        pretrained_model_path=None,
        mode="train",
        model_output_path=None,
    )
    model.load()
    model.train(*sample_data)
    return model


@pytest.fixture
def random_forest_stored_model(sample_data):
    model_path = pytest._test_path / "_data" / "random_forest_model.pkl"

    model = RegressorModel(
        model_type="random_forest",
        pretrained_model_path=None,
        mode="train",
        model_output_path=model_path,
    )
    model.load()
    model.train(*sample_data)
    model.save()

    yield model_path

    if os.path.exists(model_path):
        os.remove(model_path)


@pytest.fixture
def xgboost_stored_model(sample_data):
    model_path = pytest._test_path / "_data" / "xgboost_model.json"

    model = RegressorModel(
        model_type="xgboost",
        pretrained_model_path=None,
        mode="train",
        model_output_path=model_path,
    )
    model.load()
    model.train(*sample_data)
    model.save()

    yield model_path

    if os.path.exists(model_path):
        os.remove(model_path)
