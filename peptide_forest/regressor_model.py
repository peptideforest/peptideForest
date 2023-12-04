import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from peptide_forest import knowledge_base


class RegressorModel:
    """Regressor model for peptide forest. Wrapping aroud XGBoost and scikit learn to provide
    a unified interface for both models.

    Args:
        model_type (str): type of model to be used, either 'random_forest' or 'xgboost'
        pretrained_model_path (str): path to pretrained model
        mode (str): mode in which the model is used, either 'train', 'finetune', 'eval' or 'prune'
        additional_estimators (int): number of estimators to be added to a pretrained model
        model_output_path (str): path to store the trained model
        initial_estimators (int): number of estimators to be used for training the model
    """

    def __init__(
        self,
        model_type="random_forest",
        pretrained_model_path=None,
        mode="train",
        additional_estimators=0,
        model_output_path=None,
        initial_estimators=None,
    ):
        self.model_type = model_type
        self.pretrained_model_path = pretrained_model_path
        self.mode = mode
        self.additional_estimators = additional_estimators
        self.model_output_path = model_output_path

        self.hyperparameters = knowledge_base.parameters[
            f"hyperparameters_{model_type}"
        ]
        self.hyperparameters["n_jobs"] = mp.cpu_count() - 1

        self.initial_estimators = initial_estimators

        self._validate_str_arg(model_type, ["random_forest", "xgboost"], "Model Type")
        self._validate_str_arg(mode, ["eval", "finetune", "train", "prune"], "Mode")

        self.regressor = None

    @staticmethod
    def _validate_str_arg(arg_value, allowed_values, name):
        if arg_value in allowed_values:
            pass
        else:
            raise ValueError(
                f"{name} {arg_value} does not exist. Use one of {allowed_values}"
            )

    def score_psms(self, data):
        """Apply scoring function to classifier prediction.

        Args:
            clf (sklearn.ensemble.RandomForestRegressor): trained classifier
            data (array): input data for prediction

        Returns:
            data (array): predictions with applied scoring function
        """
        if self.model_type == "xgboost":
            dmatrix_data = xgb.DMatrix(data)
            predictions = self.regressor.predict(dmatrix_data)
        elif self.model_type == "random_forest":
            predictions = self.regressor.predict(data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return 2 * (0.5 - predictions)

    def _get_regressor(self, model_path=None):
        """Initialize random forest regressor.

        Args:
            model_path (str): path to model to load

        Returns:
            clf (sklearn.ensemble.RandomForestRegressor): classifier with added method to score PSMs
        """
        if self.initial_estimators is not None:
            self.hyperparameters["n_estimators"] = self.initial_estimators

        if self.model_type == "random_forest":
            hyperparameters = self.hyperparameters
            hyperparameters["warm_start"] = True
            clf = RandomForestRegressor(**hyperparameters)
        elif self.model_type == "xgboost":
            clf = None
        else:
            raise ValueError(
                f"Model type {self.model_type} does not exist, use either"
                f"'random_forest' or 'xgboost'."
            )

        # load model if path is given
        if model_path is not None:
            if self.model_type == "random_forest":
                clf = pickle.load(open(model_path, "rb"))
            elif self.model_type == "xgboost":
                clf = xgb.Booster()
                clf.load_model(model_path)

        return clf

    def load(self):
        """Load a regressor model that will be used for training.

        Returns:
            None

        """
        if self.mode in ["finetune", "prune"]:
            if self.pretrained_model_path is None:
                raise ValueError(
                    "pretrained_model_path has not been set, model cannot be loaded."
                )
            if self.model_type == "xgboost":
                self.regressor = self._get_regressor(
                    model_path=self.pretrained_model_path,
                )
                self.hyperparameters["n_estimators"] = self.additional_estimators
            elif self.model_type == "random_forest":
                rf_clf = self._get_regressor(
                    model_path=self.pretrained_model_path,
                )
                rf_clf.set_params(
                    n_estimators=rf_clf.n_estimators + self.additional_estimators
                )
                self.regressor = rf_clf
            else:
                raise ValueError(
                    f"Model type {self.model_type} is not implemented, use"
                    f" either 'random_forest' or 'xgboost'."
                )
        elif self.mode == "eval":
            if self.pretrained_model_path is None:
                raise ValueError(
                    "pretrained_model_path has not been set, model cannot be loaded."
                )
            self.regressor = self._get_regressor(
                model_path=self.pretrained_model_path,
            )
        elif self.mode == "train":
            self.regressor = self._get_regressor(model_path=None)
        else:
            raise ValueError(
                f"Unknown mode {self.mode}. Use one of: 'finetune', 'eval', 'train'"
            )

    def train(self, X, y):
        """Fits a regressor.

        Args:
            X: features
            y: labels

        Returns:
            None

        """
        if self.mode == "eval":
            logger.info("Model is running in eval mode, data will not be fitted.")
        elif self.mode in ["train", "finetune"]:
            if self.model_type == "random_forest":
                self.regressor.fit(X=X, y=y)
            elif self.model_type == "xgboost":
                dtrain = xgb.DMatrix(X, label=y)
                params = self.hyperparameters.copy()
                del params["n_estimators"]
                params["n_jobs"] = None
                self.regressor = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.hyperparameters["n_estimators"],
                    xgb_model=self.regressor,
                )
            else:
                raise ValueError(
                    f"Model type {self.model_type} is not implemented, use either "
                    f"'random_forest' or 'xgboost'."
                )
        elif self.mode == "prune":
            if self.model_type != "xgboost":
                raise NotImplementedError(
                    f"Pruning is currently only supported for "
                    f"models of type xgboost. Your model type is "
                    f"{self.model_type}"
                )
            self.prune_model(X=X, y=y)
        else:
            raise ValueError(
                f"Unknown mode {self.mode}. Use one of: 'finetune', 'eval', 'train'"
            )

    def get_feature_importances(self):
        """Get feature importances from trained classifier.

        Returns:
            feature_importances (np.array): feature importances
        """
        if self.model_type == "random_forest":
            return self.regressor.feature_importances_
        elif self.model_type == "xgboost":
            b = self.regressor
            score = b.get_score(importance_type="gain")  # 'weight' for gblinear
            if b.feature_names is None:
                feature_names = [f"f{i}" for i in range(b.num_features)]
            else:
                feature_names = b.feature_names
            all_features = [score.get(f, 0.0) for f in feature_names]
            all_features_arr = np.array(all_features, dtype=np.float32)
            total = all_features_arr.sum()
            if total == 0:
                return all_features_arr
            return all_features_arr / total

    def save(self):
        """
        Save trained classifier.

        Returns:
            None
        """
        if self.model_output_path is None:
            logger.warning(
                "No output path has been given, trained model won't be stored."
            )
            return

        # todo: use path objects in final version
        self.model_output_path = str(self.model_output_path)

        file_extension = self.model_output_path.split(".")[-1]
        if self.model_type == "xgboost":
            if file_extension == "json":
                self.regressor.save_model(self.model_output_path)
            else:
                self.model_output_path = self.model_output_path.split(".")[-2] + ".json"
                self.regressor.save_model(self.model_output_path)
                logger.warning(
                    f"Wrong file extension used {file_extension}. Model saved as .json"
                )
        elif self.model_type == "random_forest":
            if file_extension == "pkl":
                pickle.dump(self.regressor, open(self.model_output_path, "wb"))
            else:
                self.model_output_path = self.model_output_path.split(".")[-2] + ".pkl"
                pickle.dump(self.regressor, open(self.model_output_path, "wb"))
                logger.warning(
                    f"Wrong file extension used: {file_extension}. Model saved as .pkl"
                )

    @staticmethod
    def _identify_pruning_gamma(booster, X, y, tolerance=0.05):
        """Identifies a value for gamma to be used in pruning xgboost models.

        Args:
            booster (xgboost.Booster): model to be pruned later
            X (pd.DataFrame): features
            y (pd.Series): labels
            tolerance (float): degree to which the evaluation metric is allowed to get worse
                during pruning while still accepting the gamma value used

        Returns:
            optimal_gamma (float): optimal value to be used for pruning the model
        """
        n_leaves = []
        test_rmse = []

        g_vals = np.logspace(-4, 4, num=30, endpoint=True, base=10).tolist()
        g_vals = [0] + g_vals

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        optimal_gamma = 0
        max_leaf_reduction = 0
        baseline_n_leaves = booster.trees_to_dataframe().shape[0]

        for i, gamma in enumerate(g_vals):
            test_booster = booster.copy()
            pruning_result = {}
            pruned = xgb.train(
                {
                    "process_type": "update",
                    "updater": "prune",
                    "max_depth": test_booster.attr("max_depth"),
                    "gamma": gamma,
                },
                dtrain,
                num_boost_round=len(test_booster.get_dump()),
                xgb_model=test_booster,
                evals=[(dtest, "Test")],
                evals_result=pruning_result,
                verbose_eval=False,
            )
            current_rmse = pruning_result["Test"]["rmse"][-1]
            current_n_leaves = pruned.trees_to_dataframe().shape[0]

            if i == 0:
                baseline_rmse = current_rmse

            if current_rmse <= baseline_rmse + baseline_rmse * tolerance:
                baseline_rmse = current_rmse
                leaf_reduction = baseline_n_leaves - current_n_leaves
                if leaf_reduction > max_leaf_reduction:
                    max_leaf_reduction = leaf_reduction
                    optimal_gamma = gamma

            n_leaves.append(current_n_leaves)
            test_rmse.append(current_rmse)

        return optimal_gamma

    def prune_model(self, X, y, gamma_subset=0.2):
        """Reduce the complexity of a model by pruning nodes, that don't improve the loss
        more than a threshold (gamma).

        Args:
            booster (xgboost.Booster): booster that should be pruned
            X (pd.DataFrame): features
            y (pd.Series): labels
            gamma (float): value to be used for pruning, nodes that do not improve loss more
                than this value will be pruned.
            gamma_subset (float): calculating gamma using a fraction of the training
                data to increase calculation speed

        Returns:
            pruned_booster (xgboost.Booster): Booster with reduced complexity

        """
        # determine optimal gamma parameter for pruning
        X_gamma = X.copy().sample(frac=gamma_subset)
        y_gamma = y[X_gamma.index].copy()
        gamma = self._identify_pruning_gamma(self.regressor, X_gamma, y_gamma)
        self.regressor = xgb.train(
            {
                "process_type": "update",
                "updater": "prune",
                "max_depth": self.regressor.attr("max_depth"),
                "gamma": gamma,
            },
            xgb.DMatrix(X, label=y),
            num_boost_round=len(self.regressor.get_dump()),
            xgb_model=self.regressor,
        )
