import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from peptide_forest.regressor_model import RegressorModel


@pytest.fixture
def temp_model_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        yield tmp.name
    os.remove(tmp.name)


@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10)  # Sample feature data
    y = np.random.rand(100)  # Sample label data
    return X, y


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_correct_model_initialization(model_type, sample_data):
    model = RegressorModel(
        model_type=model_type,
        pretrained_model_path=None,
        mode="train",
        additional_estimators=10,
        model_output_path=None,
    )
    model.load()
    assert isinstance(model.regressor, (RandomForestRegressor, xgb.XGBRegressor))


def test_invalid_mode_with_pretrained_model():
    with pytest.raises(ValueError):
        RegressorModel(
            model_type="random_forest",
            pretrained_model_path="dummy/path",
            mode="invalid_mode",
            additional_estimators=10,
            model_output_path=None,
        )


@pytest.mark.parametrize(
    "model_type, fit_path, model_name",
    [
        (
            "random_forest",
            "sklearn.ensemble.RandomForestRegressor.fit",
            "random_forest_model.pkl",
        ),
        ("xgboost", "xgboost.XGBRegressor.fit", "xgboost_model.json"),
    ],
)
def test_eval_model_behavior(
    temp_model_file, model_type, fit_path, model_name, sample_data
):
    with mock.patch(fit_path) as mock_fit:
        model = RegressorModel(
            model_type=model_type,
            pretrained_model_path=pytest._test_path / "_data" / model_name,
            mode="eval",
            additional_estimators=None,
            model_output_path=None,
        )
        model.load()
        model.train(*sample_data)
        assert model.regressor is not None
        mock_fit.assert_not_called()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "model_type, model_name",
    [
        (
            "random_forest",
            "random_forest_model.pkl",
        ),
        ("xgboost", "xgboost_model.json"),
    ],
)
def test_finetune_model_behavior(model_type, model_name, sample_data):
    # train first model
    model1 = RegressorModel(
        model_type=model_type,
        pretrained_model_path=pytest._test_path / "_data" / model_name,
        mode="finetune",
        model_output_path=pytest._test_path
        / "_data"
        / f"output.{model_name.split('.')[-1]}",
    )
    model1.load()
    model1.train(*sample_data)
    model1.save()

    model2 = RegressorModel(
        model_type=model_type,
        pretrained_model_path=pytest._test_path
        / "_data"
        / f"output.{model_name.split('.')[-1]}",
        mode="finetune",
        additional_estimators=42,
    )
    model2.load()
    model2.train(*sample_data)

    buffered_model = pytest._test_path / "_data" / f"output.{model_name.split('.')[-1]}"
    buffered_model.unlink(missing_ok=True)
    assert model2.regressor.n_estimators == model1.regressor.n_estimators + 42


def test_default_behavior(sample_data):
    model = RegressorModel()
    model.load()
    model.train(*sample_data)
    model.save()

    assert model.model_type == "random_forest"
    assert model.pretrained_model_path is None
    assert model.mode == "train"
    assert model.additional_estimators == 0
    assert model.model_output_path is None
