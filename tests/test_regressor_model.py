from unittest import mock

import numpy as np
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from peptide_forest.regressor_model import RegressorModel


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_correct_model_initialization(model_type, sample_data):
    model = RegressorModel(
        model_type=model_type,
        pretrained_model_path=None,
        mode="train",
        model_output_path=None,
    )
    model.load()
    assert isinstance(model.regressor, (RandomForestRegressor, type(None)))


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_initial_estimators_parameter(model_type, sample_data):
    model = RegressorModel(
        model_type=model_type,
        mode="train",
        additional_estimators=10,
        initial_estimators=5,
    )
    model.load()
    model.train(*sample_data)
    if model_type == "random_forest":
        assert model.regressor.n_estimators == 5
    elif model_type == "xgboost":
        assert len(model.regressor.get_dump()) == 5


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
    "model_type, fit_path",
    [
        ("random_forest", "sklearn.ensemble.RandomForestRegressor.fit"),
        ("xgboost", "xgboost.train"),
    ],
)
def test_eval_model_behavior(
    temp_model_file,
    model_type,
    fit_path,
    sample_data,
    random_forest_stored_model,
    xgboost_stored_model,
):
    model_paths = {
        "random_forest": random_forest_stored_model,
        "xgboost": xgboost_stored_model,
    }

    with mock.patch(fit_path) as mock_fit:
        model = RegressorModel(
            model_type=model_type,
            pretrained_model_path=model_paths[model_type],
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
    "model_type",
    [("random_forest"), ("xgboost")],
)
def test_finetune_model_behavior(
    model_type, sample_data, random_forest_stored_model, xgboost_stored_model
):
    model_paths = {
        "random_forest": random_forest_stored_model,
        "xgboost": xgboost_stored_model,
    }

    # train first model
    model1 = RegressorModel(
        model_type=model_type,
        pretrained_model_path=model_paths[model_type],
        mode="finetune",
        model_output_path=pytest._test_path
        / "_data"
        / f"output.{model_paths[model_type].name.split('.')[-1]}",
    )
    model1.load()
    model1.train(*sample_data)
    model1.save()

    model2 = RegressorModel(
        model_type=model_type,
        pretrained_model_path=pytest._test_path
        / "_data"
        / f"output.{model_paths[model_type].name.split('.')[-1]}",
        mode="finetune",
        additional_estimators=42,
    )
    model2.load()
    model2.train(*sample_data)

    buffered_model = (
        pytest._test_path
        / "_data"
        / f"output.{model_paths[model_type].name.split('.')[-1]}"
    )
    buffered_model.unlink(missing_ok=True)
    if model_type == "random_forest":
        assert model2.regressor.n_estimators == model1.regressor.n_estimators + 42
    elif model_type == "xgboost":
        assert len(model2.regressor.get_dump()) == len(model1.regressor.get_dump()) + 42


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "model_type, model_name",
    [
        ("random_forest", "temp_random_forest_model.pkl"),
        ("xgboost", "temp_xgboost_model.json"),
    ],
)
def test_xgboost_base_trees_remain_same(model_type, model_name, sample_data):
    temp_model = pytest._test_path / "_data" / model_name
    model = RegressorModel(
        model_type=model_type,
        mode="train",
        model_output_path=temp_model,
    )
    model.load()
    model.train(*sample_data)
    model.save()

    ft_model = RegressorModel(
        model_type=model_type,
        mode="finetune",
        additional_estimators=40,
        pretrained_model_path=temp_model,
    )
    ft_model.load()
    ft_model.train(*sample_data)
    temp_model.unlink(missing_ok=True)

    if model_type == "xgboost":
        original_dump = model.regressor.get_dump()
        loaded_dump = ft_model.regressor.get_dump()[:100]
        for original_tree, loaded_tree in zip(original_dump, loaded_dump):
            assert original_tree == loaded_tree, "Trees should be identical"
    elif model_type == "random_forest":
        for original_tree, loaded_tree in zip(
            model.regressor.estimators_, ft_model.regressor.estimators_[:100]
        ):
            assert np.array_equal(
                original_tree.tree_.value, loaded_tree.tree_.value
            ), "Trees should be identical"


def test_get_feature_importances(sample_data):
    model = RegressorModel(model_type="random_forest")
    model.load()
    model.train(*sample_data)
    rf_feature_importances = model.get_feature_importances()

    model = RegressorModel(model_type="xgboost")
    model.load()
    model.train(*sample_data)
    xgb_feature_importances = model.get_feature_importances()

    assert isinstance(xgb_feature_importances, type(rf_feature_importances))
    assert len(xgb_feature_importances) == len(rf_feature_importances)


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


def test_prune_model_regressor_type_post_pruning(mock_regressor_model, sample_data):
    mock_regressor_model.mode = "prune"
    mock_regressor_model.train(*sample_data)
    assert isinstance(
        mock_regressor_model.regressor, xgb.Booster
    ), "Regressor should be an instance of xgboost.XGBRegressor"


def test_prune_model_reduced_complexity(mock_regressor_model, sample_data):
    original_leaf_count = mock_regressor_model.regressor.trees_to_dataframe().shape[0]

    mock_regressor_model.mode = "prune"
    mock_regressor_model.train(*sample_data)
    pruned_leaf_count = mock_regressor_model.regressor.trees_to_dataframe().shape[0]

    assert (
        pruned_leaf_count <= original_leaf_count
    ), "Pruned model should have fewer leaf nodes than the original model"


def test_prune_model_same_root_node(mock_regressor_model, sample_data):
    original_root = mock_regressor_model.regressor.trees_to_dataframe().iloc[0]

    mock_regressor_model.mode = "prune"
    mock_regressor_model.train(*sample_data)
    pruned_root = mock_regressor_model.regressor.trees_to_dataframe().iloc[0]

    assert original_root.equals(
        pruned_root
    ), "The root node of the pruned model should be the same as the original model"
