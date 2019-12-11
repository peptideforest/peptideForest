"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Define and fit classification model to target-decoy data
"""

import types

# pylint: disable=unused-import
from typing import Any, Dict, Pattern, Set, Tuple, List

import numpy as np
import pandas as pd


ml_methods = ["RF", "RF-reg", "GBT", "SVM", "GNB", "KNN", "LR", "ADA", "percolator"]


def set_random_forest(hp_dict_in: Dict = None) -> Any:
    """
  Define a random forest model
  Arguments:
    - hp_dict_in: dictionary containing hyperparameter values. These are:
        - n_estimators: number of trees in the forest, default = 100.
        - random_state: which random state to use (for reproducability, default =None)
        - class_weights: weights for each label group, as dictionary, default = None
        - max_depth: maximum depth of each tree, default = None
        - min_samples_split: min number of records needed in order to split a branch, default = 2
        - min_samples_leaf: min number of records to be left on each branch after a split,
                            default = 1
        - max_features: max number of features to use when looking for best split,
                        default = sqrt(n_features), defined as 'auto'
        - n_jobs: number of jobs to run in parallel, default is None (1 job)
  Returns:
    - clf: RF model
  """
    from sklearn.ensemble import RandomForestClassifier

    # dictionary of default hyperparameter values
    hp_dict = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": None,
        "class_weights": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",
        "n_jobs": None,
    }
    if hp_dict_in:
        hp_dict.update(hp_dict_in)

    clf = RandomForestClassifier(
        n_estimators=hp_dict["n_estimators"],
        max_depth=hp_dict["max_depth"],
        random_state=hp_dict["random_state"],
        class_weight=hp_dict["class_weights"],
        min_samples_split=hp_dict["min_samples_split"],
        min_samples_leaf=hp_dict["min_samples_leaf"],
        max_features=hp_dict["max_features"],
        n_jobs=hp_dict["n_jobs"],
    )
    return clf


def set_random_forest_reg(hp_dict_in: Dict = None) -> Any:
    """
  Define a random forest regression model
  Arguments:
    - hp_dict_in: dictionary containing hyperparameter values. These are:
        - n_estimators: number of trees in the forest, default = 100.
        - random_state: which random state to use (for reproducability, default =None)
        - max_depth: maximum depth of each tree, default = None
        - min_samples_split: min number of records needed in order to split a branch, default = 2
        - min_samples_leaf: min number of records to be left on each branch after a split,
                            default = 1
        - max_features: max number of features to use when looking for best split,
                        default = sqrt(n_features), defined as 'auto'
        - n_jobs: number of jobs to run in parallel, default is None (1 job)
  Returns:
    - clf: RF model, regressor
  """
    from sklearn.ensemble import RandomForestRegressor

    # dictionary of default hyperparameter values
    hp_dict = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",
        "n_jobs": None,
    }
    if hp_dict_in:
        hp_dict.update(hp_dict_in)

    clf = RandomForestRegressor(
        n_estimators=hp_dict["n_estimators"],
        max_depth=hp_dict["max_depth"],
        random_state=hp_dict["random_state"],
        min_samples_split=hp_dict["min_samples_split"],
        min_samples_leaf=hp_dict["min_samples_leaf"],
        max_features=hp_dict["max_features"],
        n_jobs=hp_dict["n_jobs"],
    )
    return clf


def set_gradient_boosted_trees(hp_dict_in: Dict = None) -> Any:
    """
  Define a gradient boosted trees model
  Arguments:
    - hp_dict_in: dictionary containing hyperparameter values. These are:
        - n_estimators: number of trees in the forest, default = 100.
        - random_state: which random state to use (for reproducability, default =None)
        - class_weights: weights for each label group, as dictionary, default = None
        - max_depth: maximum depth of each tree, default = None
        - min_samples_split: min number of records needed in order to split a branch, default = 2
        - min_samples_leaf: min number of records to be left on each branch after a split,
                            default = 1
        - max_features: max number of features to use when looking for best split,
                        default = sqrt(n_features), defined as 'auto'
  Returns:
    - clf: GBT model
  """
    from sklearn.ensemble import GradientBoostingClassifier

    # dictionary of default hyperparameter values
    hp_dict = {
        "n_estimators": 100,
        "max_depth": 1,
        "random_state": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",
    }
    if hp_dict_in:
        hp_dict.update(hp_dict_in)

    clf = GradientBoostingClassifier(
        n_estimators=hp_dict["n_estimators"],
        max_depth=hp_dict["max_depth"],
        random_state=hp_dict["random_state"],
        min_samples_split=hp_dict["min_samples_split"],
        min_samples_leaf=hp_dict["min_samples_leaf"],
        max_features=hp_dict["max_features"],
    )
    return clf


def set_ada_dec_tree() -> Any:
    """
  Define an ada boosted classifier with decision tree
  Returns:
    - clf: ada boosted classifier with decision tree
  """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200
    )
    return clf


def set_support_vector_machine(hp_dict_in: Dict = None) -> Any:
    """
  Define a linear support vector machine
  Returns:
    - clf: support vector machine model
  """
    from sklearn import svm

    hp_dict = {"max_iter": 2000, "random_state": None, "tol": 1e-4}

    if hp_dict_in:
        hp_dict.update(hp_dict_in)

    clf = svm.LinearSVC(
        max_iter=hp_dict["max_iter"], random_state=hp_dict["random_state"]
    )
    return clf


def set_gaussian_naive_bayes() -> Any:
    """
  Define a Gaussian Naive Bayes classifier
  Returns:
    - clf: Gaussian Naive Bayes classifier model
  """
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    return clf


def set_knn() -> Any:
    """
  Define a K nearest neighbors classifier
  Returns:
    - clf: K nearest neighbors classifier model
  """
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=7)
    return clf


def set_logistic_regression() -> Any:
    """
  Define a logistic regression classifier
  Returns:
    - clf: logistic regression classifier model
  """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(solver="saga")
    return clf


def get_training_model(training_type: str = "RF", hp_dict_in: Dict = None) -> Any:
    """
  Select the particular model to use for training.
  Arguments:
    - training_type: which type of model to use. Options include:
      - 'RF': 'Random Forest',
      - 'RF-reg': 'Random Forest - regression',
      - 'GBT': 'Gradient Boosted Trees',
      - 'SVM': 'Support Vector Machine',
      - 'GNB': 'Gaussian Naive Bayes classifier',
      - 'KNN': 'K Nearest Neighbors',
      - 'LR': 'Logistic Regression',
      - 'ADA': 'Ada Boosted Classifier with Decision Trees'
    - hp_dict_in: dictionary containing hyperparmeters for the model, which are the same as the
                  scikit learn parameters.
  Returns:
    - clf: selected classifier with hyperparameters and scoring_func function added (for scoring
           a PSM based on if it is a top-target)
  """

    training_types = {
        "RF": "Random Forest",
        "RF-reg": "Random Forest - regression",
        "GBT": "Gradient Boosted Trees",
        "SVM": "Support Vector Machine",
        "GNB": "Gaussian Naive Bayes classifier",
        "KNN": "K Nearest Neighbors",
        "LR": "Logistic Regression",
        "ADA": "Ada Boosted Classifier with Decision Trees",
    }

    if training_type not in training_types.keys():
        print(f"{training_type} not supported. Select one of: ")
        print("\n".join("{}:- for {}".format(k, v) for k, v in training_types.items()))
        return None

    if training_type == "RF":
        # set model to be Random Forest Classifier
        clf = set_random_forest(hp_dict_in)
    if training_type == "RF-reg":
        # set model to be Random Forest Regression
        clf = set_random_forest_reg(hp_dict_in)
    if training_type == "GBT":
        # set model to be Gradient Boosted Trees Classifier
        clf = set_gradient_boosted_trees(hp_dict_in)
    elif training_type == "SVM":
        # set model to be linear support vector machine
        clf = set_support_vector_machine(hp_dict_in)
    elif training_type == "GNB":
        # set model to be a gaussian naive bayes
        clf = set_gaussian_naive_bayes()
    elif training_type == "KNN":
        # set model to be K nearest neighbors
        clf = set_knn()
    elif training_type == "LR":
        # set model to be logistic regression
        clf = set_logistic_regression()
    elif training_type == "ADA":
        # set model to be ada bossted classifier
        clf = set_ada_dec_tree()

    def score_psm(clf: Any, data: Any) -> Any:
        if training_type in ["RF", "GNB", "KNN", "LR"]:
            scoring_func = [c[0] for c in clf.predict_proba(data)]
        elif training_type in ["SVM", "ADA", "GBT"]:
            scoring_func = [-c for c in clf.decision_function(data)]
        elif training_type in ["RF-reg"]:
            scoring_func = [2.0 * (-c + 0.5) for c in clf.predict(data)]
        return scoring_func

    clf.score_psm = types.MethodType(score_psm, clf)

    return clf
