import multiprocessing as mp
import pickle

import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from peptide_forest import knowledge_base


class RegressorModel:
    def __init__(
        self,
        model_type,
        pretrained_model_path,
        mode,
        additional_estimators,
        model_output_path,
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

        self._validate_str_arg(model_type, ["random_forest", "xgboost"], "Model Type")
        self._validate_str_arg(mode, ["eval", "finetune", "train"], "Mode")

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
        return 2 * (0.5 - self.regressor.predict(data))

    def _get_regressor(self, model_path=None):
        """Initialize random forest regressor.

        Args:
            model_path (str): path to model to load

        Returns:
            clf (sklearn.ensemble.RandomForestRegressor): classifier with added method to score PSMs
        """
        if self.model_type == "random_forest":
            clf = RandomForestRegressor(**self.hyperparameters)
        elif self.model_type == "xgboost":
            clf = xgb.XGBRegressor(**self.hyperparameters)

        # load model if path is given
        if model_path is not None:
            if self.model_type == "random_forest":
                clf = pickle.load(open(model_path, "rb"))
            elif self.model_type == "xgboost":
                clf.load_model(model_path)

        return clf

    def load(self):
        if self.mode == "finetune" and self.pretrained_model_path is not None:
            _clf = self._get_regressor(
                model_path=self.pretrained_model_path,
            )
            self.hyperparameters = _clf.get_params()
            self.hyperparameters["n_estimators"] += self.additional_estimators
        self.regressor = self._get_regressor(
            model_path=self.pretrained_model_path,
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
            return

        if self.model_type == "random_forest":
            self.regressor.fit(X=X, y=y)
        elif self.model_type == "xgboost":
            self.regressor.fit(X=X, y=y, xgb_model=self.pretrained_model_path)

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

        file_extension = self.model_output_path.split(".")[-1]
        if self.model_type == "xgboost":
            if file_extension == "json":
                self.regressor.save_model(self.model_output_path)
            else:
                self.model_output_path = self.model_output_path.split(".")[0] + ".json"
                self.regressor.save_model(self.model_output_path)
                logger.warning(
                    f"Wrong file extension used {file_extension}. Model saved as .json"
                )
        elif self.model_type == "random_forest":
            if file_extension == "pkl":
                pickle.dump(self.regressor, open(self.model_output_path, "wb"))
            else:
                self.model_output_path = self.model_output_path.split(".")[0] + ".pkl"
                pickle.dump(self.regressor, open(self.model_output_path, "wb"))
                logger.warning(
                    f"Wrong file extension used: {file_extension}. Model saved as .pkl"
                )
