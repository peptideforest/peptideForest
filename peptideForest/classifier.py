# [TRISTAN] remove all not needed?
ml_methods = ["RF", "RF-reg", "GBT", "SVM", "GNB", "KNN", "LR", "ADA", "percolator"]


def set_random_forest(hp_dict_in=None):
    """
    Define a random forest model.
    Args:
        hp_dict_in (Dict, optional): dictionary containing hyperparameter values

    Returns:
        clf (Any): RF model
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


def set_random_forest_reg(hp_dict_in=None):
    """
    Define a random forest regression model.
    Args:
        hp_dict_in (Dict, optional): dictionary containing hyperparameter values

    Returns:
        clf (Any): RF model, regressor
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


def set_gradient_boosted_trees(hp_dict_in=None):
    """
    Define a gradient boosted trees model.
    Args:
        hp_dict_in (Dict, optional): dictionary containing hyperparameter values

    Returns:
        clf (Any): GBT model
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


def set_ada_dec_tree():
    """
    Define an ada boosted classifier with decision tree.
    Returns:
        clf (Any): ada boosted classifier with decision tree
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200
    )
    return clf


def set_support_vector_machine(hp_dict_in=None):
    """
    Define a linear support vector machine.
    Args:
        hp_dict_in (Dict, optional): dictionary containing hyperparameter values

    Returns:
        clf (Any): support vector machine model
    """
    from sklearn import svm

    hp_dict = {"max_iter": 2000, "random_state": None, "tol": 1e-4}

    if hp_dict_in:
        hp_dict.update(hp_dict_in)

    clf = svm.LinearSVC(
        max_iter=hp_dict["max_iter"], random_state=hp_dict["random_state"]
    )
    return clf


def set_gaussian_naive_bayes():
    """
    Define a Gaussian Naive Bayes classifier.
    Returns:
        clf (Any): Gaussian Naive Bayes model
    """
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    return clf


def set_knn():
    """
    Define a K nearest neighbours classifier.
    Returns:
        clf (Any): K nearest neighbours classifier model
    """
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=7)
    return clf


def set_logistic_regression():
    """
    Define a logistic regression classifier.
    Returns:
        clf (Any): logistic regression classifier model
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(solver="saga")
    return clf
