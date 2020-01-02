from peptideForest import classifier, setup_dataset

import warnings
import types
import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing
from sklearn.exceptions import DataConversionWarning


# [TRISTAN] rare case where default set for get_fdr
def get_q_vals(df, score_col, frac_tp, top_psm_only, get_fdr=True):
    """
    Calculate q-value for each PSM based on the score in the column score_col
    Args:
        df (pd.DataFrame): experiment dataframe
        score_col (str): name of column to rank PSMs by
        frac_tp (float): estimate of fraction of true positives in target dataset
        # [TRISTAN] ist fix true maybe remove?
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        # [TRISTAN] ist fix false maybe remove?
        get_fdr (bool): if True return the false detection rate as q-value, else return q-values

    Returns:
        df_scores (pd.DataFrame): dataframe containing q-values, score and whether PSM is target or decoy
    """
    engines = [c for c in df.columns if "Score_processed" in c]
    engines = [
        e
        for e in engines
        if e not in [f"Score_processed_{m}" for m in classifier.ml_methods]
    ]
    engines = [e.split("Score_processed_")[-1] for e in engines]

    # [TRISTAN] from_method als argument gekürzt
    df_scores = df.sort_values(score_col, ascending=False).copy(deep=True)

    df_scores = find_psms_to_keep(df_scores, score_col)
    df_scores = df_scores[df_scores["keep in"]]

    if top_psm_only:
        df_scores = df_scores.drop_duplicates("Spectrum ID")

    frac_tp = frac_tp * sum(~df_scores["Is decoy"]) / sum(df_scores["Is decoy"])
    df_scores = df_scores[["Spectrum ID", "Sequence", score_col, "Is decoy"]]
    target_decoy_dummies = pd.get_dummies(df_scores["Is decoy"])
    df_scores.loc[
        :, str(target_decoy_dummies.columns[0])
    ] = target_decoy_dummies.values[:, 0]
    df_scores.loc[
        :, str(target_decoy_dummies.columns[1])
    ] = target_decoy_dummies.values[:, 1]
    df_scores = df_scores.rename({"False": "Target", "True": "Decoy"}, axis=1)
    df_scores["FDR"] = (
        frac_tp * df_scores["Decoy"].cumsum() / df_scores["Target"].cumsum()
    )
    if get_fdr:
        df_scores["q-value"] = df_scores["FDR"]
    else:
        df_scores["q-value"] = df_scores["FDR"].cummax()

    return df_scores


def get_train_test_sets(
    df, use_cross_validation,
):
    """
    Split dataframe into multiple sets for training.
    Args:
        df (pd.DataFrame): experiment dataframe
        use_cross_validation (bool): if True use 3-fold cross-validation, if False train on all targets

    Returns:
        training_data (pd.DataFrame):   list containing data splits. If use_cross_validation is True contains has
                                        three cv splits. If use_cross_validation is False, contains train split
                                        (all targets, 50% decoys) and test split (all targets, other 50% decoys)
    """
    # Start a list to contain the data
    training_data = []
    # if use_cv, split data into 3 equal sets of spectrum IDs
    if use_cross_validation:
        spec_ids = df["Spectrum ID"].unique()
        np.random.shuffle(spec_ids)
        div_by_three_len = 3 * int(np.floor(len(spec_ids) / 3))
        spec_ids_list = np.split(spec_ids[:div_by_three_len], 3)
        spec_ids_list[0] = np.append(spec_ids_list[0], spec_ids[div_by_three_len:])
        for spec_ids in spec_ids_list:
            training_data.append(df[df["Spectrum ID"].isin(spec_ids)].copy(deep=True))
    else:
        train_decoys, test_decoys = model_selection.train_test_split(
            df[df["Is decoy"]], test_size=0.5, random_state=0
        )
        train = pd.concat([df[~df["Is decoy"]], train_decoys]).sample(frac=1)
        training_data.append(train)
        test = pd.concat([df[~df["Is decoy"]], test_decoys]).sample(frac=1)
        training_data.append(test)

    return training_data


def find_psms_to_keep(df, score_col):
    """
    Flag PSMs to remove from q-value calculations, when the highest score for a given spectrum is shared by
    two or more PSMs.
    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): score column to use when comparing PSM scores

    Returns:
        df (pd.DataFrame): updated dataframe with flag 'keep in': False means remove when calculating q-values
    """
    max_col = "max_" + score_col.split("Score_processed_")[-1]
    eq_col = "eq_" + score_col.split("Score_processed_")[-1]

    # Get the maximum value for each spectrum ID
    df.loc[:, max_col] = df.groupby("Spectrum ID")[score_col].transform("max")
    # Replace 0 with nan, as when the maximum score is 0, there are no PSMs
    df.loc[:, max_col] = df[max_col].replace(0.0, np.nan)
    df.loc[:, eq_col] = df[max_col] == df[score_col]

    # Flag to keep psm in set
    df["keep in"] = True

    # Number PSMs with max score for each spectrum
    num_with_top_val = df.groupby("Spectrum ID")[eq_col].sum()
    # Index of spectra where more than one psm has the max score
    inds = num_with_top_val.index[num_with_top_val > 1]
    # Get dataframe for those spectra and take only psms that have the max score
    df_with_top_val = df[df["Spectrum ID"].isin(inds) & df[eq_col]].copy(deep=True)
    # Check how many decoys there are for each spectra
    df_num_decoys = df_with_top_val.groupby("Spectrum ID")["Is decoy"].agg(
        ["count", "sum", "nunique"]
    )
    # Get spectra where all PSMs with max_score should be dropped
    # Target and decoy are ranked top
    inds_drop = set(df_num_decoys[df_num_decoys["nunique"] == 2].index)
    # Only targets are top
    inds_drop = inds_drop.union(set(df_num_decoys[df_num_decoys["sum"] == 0].index))

    # Get spectra where there are only decoys with the max score, one of which will be kept
    inds_keep = df_num_decoys[df_num_decoys["sum"] == df_num_decoys["count"]].index

    # Get list of PSMs to drop: choose one at random to keep when both were decoys
    psms_keep = list(
        df[df["Spectrum ID"].isin(inds_keep) & df[eq_col]]
        .sample(frac=1.0)
        .drop_duplicates("Spectrum ID")
        .index
    )

    # And drop the other one
    psms_drop = list(
        df[
            df["Spectrum ID"].isin(inds_keep) & df[eq_col] & (~df.index.isin(psms_keep))
        ].index
    )

    # Add those that where all PSMs with max_score should be dropped
    # Drop all PSMs for those spectrums
    psms_drop += list(df[df["Spectrum ID"].isin(inds_drop)].index)

    # Change to False if in psms_drop
    df.loc[df.index.isin(psms_drop), "keep in"] = False
    # Remove columns that are no longer needed
    df = df.drop([max_col, eq_col], axis=1)

    return df


def calc_num_psms(
    df, score_col, q_cut, frac_tp,
):
    """
    Calculate the number of PSMs with q-value < 1%, using the following criteria:
        - only the highest ranked PSMs for each spectra are considered
        - redundant (i.e. duplicate) peptide sequences are discarded
        - only target PSMs are counted
    Args:
        df (pd.DataFrame): dataframe containing scores for each PSM
        score_col (str): name of column to rank PSMs by
        q_cut (float): cut off for q-values below which a target PSM is counted as top-target
        frac_tp (float): estimate of fraction of true positives in target dataset

    Returns:
        n_psms (int): number of PSMs with q-value < 1% that match the above criteria

    """
    # Get the q-values
    df_scores_new = get_q_vals(df, score_col, frac_tp=frac_tp, top_psm_only=True)
    # Order by the q-value, and keep only the first PSM for each spectra
    df_scores_sub = df_scores_new.sort_values("q-value", ascending=True)
    df_scores_sub = df_scores_sub.drop_duplicates("Spectrum ID")
    # Get the number of target PSMs with q-value < 1%
    n_psms = len(
        df_scores_sub.loc[
            (df_scores_sub["q-value"] <= q_cut) & (~df_scores_sub["Is decoy"]), :
        ]
    )
    return n_psms


def get_top_targets(df, score_col, q_cut, frac_tp):
    """
    Return only target PSMs with q-value less than q_cut.
    Args:
        df (pd.DataFrame): dataframe containing experimental data
        score_col (str): name of column containing score to rank PSMs by
        q_cut (float): cut off for q-values below which a target PSM is counted as top-target
        frac_tp (float): estimate of fraction of true positives in target dataset

    Returns:
        df.loc[inds, :]: inds is the index of PSMs matching criteria
    """
    df_scores = get_q_vals(
        df, score_col, frac_tp=frac_tp, top_psm_only=False, get_fdr=False
    )
    inds = df_scores[(df_scores["q-value"] <= q_cut) & (~df_scores["Is decoy"])].index
    return df.loc[inds, :]


def get_train_set(
    df, score_col, q_cut, frac_tp, train_top_data, sample_frac=1.0,
):
    """
    Return training dataset containing sample of decoys and all target PSMs with q-value less than 1%
    Args:
        df (pd.DataFrame): dataframe containing experiment data
        score_col (str): name of column containing score to rank PSMs by
        q_cut (float): cut off for q-values below which a target PSM is counted as a top-target
        frac_tp (float): estimate of fraction of true positives in target dataset
        train_top_data (bool): if False train on all the data, True only train on top target/decoy per spectrum
        # [TRISTAN] if to be included muss das noch nach oben vererbt werden
        sample_frac (float): ratio of decoy PSMs to target and decoy PSMs

    Returns:
        train (pd.DataFrame): dataframe containing selected target and decoy PSMs
    """
    # [TRISTAN] prep.get_top_target_decoy(df, score_col=score_col)
    if train_top_data:
        # Take only the top target and top decoy for each spectrum
        df_to_train = setup_dataset.get_top_target_decoy(df, score_col=score_col)
    else:
        df_to_train = df.copy(deep=True)
    train_targets = get_top_targets(
        df_to_train, score_col, q_cut=q_cut, frac_tp=frac_tp
    )
    n_sample = int(sample_frac * len(train_targets))
    if n_sample > len(df_to_train[df_to_train["Is decoy"]]):
        train_decoys = df_to_train[df_to_train["Is decoy"]].sample(
            n=n_sample, replace=True
        )
    else:
        train_decoys = df_to_train[df_to_train["Is decoy"]].sample(n=n_sample)
    # Concatenate and reorder
    train = pd.concat([train_targets, train_decoys]).sample(frac=1)
    return train


# [TRISTAN] This function was moved here from setup_dataset
def replace_missing_data_cv(train_j, train_k, test):
    """
    Replace missing delta_scores with minimum value from training data.
    Args:
        train_j (pd.DataFrame): dataframe containing feature values for training split
        train_k (pd.DataFrame): dataframe containing feature values for training split
        test (pd.DataFrame): dataframe containing feature values for test split

    Returns:
        train_j (pd.DataFrame): input dataframe with missing values filled in
        train_k (pd.DataFrame): input dataframe with missing values filled in
        test (pd.DataFrame): input dataframe with missing values filled in
    """
    del_score_cols = [f for f in train_j.columns if "delta_score" in f]
    for del_score_col in del_score_cols:
        min_val = min(train_j[del_score_col].min(), train_k[del_score_col].min())
        train_j[del_score_col] = train_j[del_score_col].fillna(min_val)
        train_k[del_score_col] = train_k[del_score_col].fillna(min_val)
        test[del_score_col] = test[del_score_col].fillna(min_val)

    return train_j, train_k, test


def replace_missing_data_top_targets(training_data):
    """
    Replace missing delta_scores with minimum value from training data.
    Args:
        training_data (List): list containing data splits, 0 is training data, 1 is test data

    Returns:
        training_data (List): input list with missing values filled in
    """
    del_score_cols = [f for f in training_data[0].columns if "delta_score" in f]
    for del_score_col in del_score_cols:
        min_val = training_data[0][del_score_col].min()
        training_data[0][del_score_col] = training_data[0][del_score_col].fillna(
            min_val
        )
        training_data[1][del_score_col] = training_data[1][del_score_col].fillna(
            min_val
        )

    return training_data


def get_training_model(training_type="RF", hp_dict_in=None):
    """
    Select the particular model to use for training.
    Args:
        training_type (str, optional): type of model to use. Available options:
            - 'RF': 'Random Forest',
            - 'RF-reg': 'Random Forest - regression',
            - 'GBT': 'Gradient Boosted Trees',
            - 'SVM': 'Support Vector Machine',
            - 'GNB': 'Gaussian Naive Bayes classifier',
            - 'KNN': 'K Nearest Neighbors',
            - 'LR': 'Logistic Regression',
            - 'ADA': 'Ada Boosted Classifier with Decision Trees'
        hp_dict_in (Dict, optional):    dictionary containing hyper parameters for the model, which are the same
                                        as the scikit-learn parameters.

    Returns:
        clf (Any):  selected classifier with hyper parameters and scoring_func added (for scoring a PSM based
                    on whether it is a top-target)

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

    # [TRISTAN] hier ValueError raisen?
    if training_type not in training_types.keys():
        print(f"{training_type} not supported. Select one of: ")
        print("\n".join("{}:- for {}".format(k, v) for k, v in training_types.items()))
        return None

    if training_type == "RF":
        # set model to be Random Forest Classifier
        clf = classifier.set_random_forest(hp_dict_in)
    if training_type == "RF-reg":
        # set model to be Random Forest Regression
        clf = classifier.set_random_forest_reg(hp_dict_in)
    if training_type == "GBT":
        # set model to be Gradient Boosted Trees Classifier
        clf = classifier.set_gradient_boosted_trees(hp_dict_in)
    elif training_type == "SVM":
        # set model to be linear support vector machine
        clf = classifier.set_support_vector_machine(hp_dict_in)
    elif training_type == "GNB":
        # set model to be a gaussian naive bayes
        clf = classifier.set_gaussian_naive_bayes()
    elif training_type == "KNN":
        # set model to be K nearest neighbors
        clf = classifier.set_knn()
    elif training_type == "LR":
        # set model to be logistic regression
        clf = classifier.set_logistic_regression()
    elif training_type == "ADA":
        # set model to be ada bossted classifier
        clf = classifier.set_ada_dec_tree()

    def score_psm(clf, data):
        if training_type in ["RF", "GNB", "KNN", "LR"]:
            scoring_func = [c[0] for c in clf.predict_proba(data)]
        elif training_type in ["SVM", "ADA", "GBT"]:
            scoring_func = [-c for c in clf.decision_function(data)]
        elif training_type in ["RF-reg"]:
            scoring_func = [2.0 * (-c + 0.5) for c in clf.predict(data)]
        return scoring_func

    clf.score_psm = types.MethodType(score_psm, clf)

    return clf


def fit_model_cv(
    df_training,
    classifier,
    n_train,
    train_top_data,
    feature_cols,
    hyper_parameters,
    q_cut_train,
    frac_tp,
    feature_importance,
    clfs,
    i_it,
    score_col,
    training_data,
    old_cols,
):
    """
    Fit model_at.
    Args:
        df_training (pd.DataFrame): dataframe containing top target and top decoy for each spectrum
        classifier (str): which model to fit
        n_train (int): number of training iterations
        train_top_data (bool): if True only train on top target/decoy per spectrum, if False train on all data
        feature_cols (List): list of features to use when fitting the model
        hyper_parameters (Dict): dictionary containing hyper parameters for the model
        q_cut_train (float):    cut off for q-values below which a target PSM is counted as top-target,
                                when setting top-targets to train on
        frac_tp (float): estimate of fraction of true positives in target dataset
        training_data (pd.DataFrame):   list containing data splits. Contains train split (all targets, 50% decoys)
                                        and test split (all targets, other 50% decoys)
        score_col (str): particular original score
        i_it (int): current iteration cycle
        clfs (Dict): dictionary containing a list of classifiers
        old_cols (List): original column names
        feature_importance (List): list containing feature importance
    Returns:
        clfs (Dict):    dictionary containing a list of classifiers for each initial_score_col, that are fitted
                        to data at each iteration, with scalers for normalizing data
        df_training (pd.DataFrame): input dataframe with score column added
        feature_importance (Dict): dictionary of dataframe containing feature importance for each initial_score_col
    """

    # Fit the model to the data
    clfs_sub = []
    df_training.loc[:, "model_score_train"] = [0 for _ in np.arange(len(df_training))]
    for it in np.arange(len(training_data)):
        j, k = [s for s in [0, 1, 2] if s != it]
        # copy the data (so that the scaling of the data does not change the original)
        test = training_data[it].copy(deep=True)
        train_j = training_data[j][list(old_cols.union([score_col]))].copy(deep=True)
        train_k = training_data[k][list(old_cols.union([score_col]))].copy(deep=True)

        # Fill in missing values
        train_j, train_k, test = replace_missing_data_cv(train_j, train_k, test)

        # Train on 2/3 data, with top targets and 50% decoys
        train = pd.concat([train_j, train_k])
        train = get_train_set(
            train,
            score_col,
            frac_tp=frac_tp,
            train_top_data=train_top_data,
            q_cut=q_cut_train,
        )

        # Scale the data
        scaler = preprocessing.StandardScaler().fit(train.loc[:, feature_cols])
        train.loc[:, feature_cols] = scaler.transform(train.loc[:, feature_cols])
        train_j.loc[:, feature_cols] = scaler.transform(train_j.loc[:, feature_cols])
        train_k.loc[:, feature_cols] = scaler.transform(train_k.loc[:, feature_cols])
        test.loc[:, feature_cols] = scaler.transform(test.loc[:, feature_cols])

        # Get the classifier
        clf = get_training_model(classifier, hyper_parameters)
        clf.fit(train[feature_cols], train["Is decoy"])
        clfs_sub.append([clf, scaler])

        # Add the feature importance
        if i_it >= n_train:
            if classifier.upper() in ["RF", "RF-REG", "ADA", "GBT"]:
                feature_importance_sub = clf.feature_importances_
            elif classifier.upper() in ["SVM", "LR"]:
                feature_importance_sub = clf.coef_[0]
            elif classifier.upper() in ["KNN", "GNB"]:
                feature_importance_sub = [0 for _ in range(len(feature_cols))]
            feature_importance.append(feature_importance_sub)

        # Score each of the splits
        training_data[it].loc[:, "model_score"] = clf.score_psm(test[feature_cols])
        training_data[j].loc[:, "model_score_train"] = clf.score_psm(
            train_j[feature_cols]
        )
        training_data[k].loc[:, "model_score_train"] = clf.score_psm(
            train_k[feature_cols]
        )
        df_training.loc[training_data[it].index, "model_score"] = training_data[it][
            "model_score"
        ]

        # Get the training score, which we take as the average score over each time the
        # PSM is trained on
        df_training.loc[train_j.index, "model_score_train"] += training_data[j][
            "model_score_train"
        ].values
        df_training.loc[train_k.index, "model_score_train"] += training_data[k][
            "model_score_train"
        ].values
    df_training.loc[:, "model_score_train"] = df_training["model_score_train"] / (
        len(training_data) - 1
    )
    clfs.append(clfs_sub)

    return clfs, df_training, feature_importance


def fit_model_at(
    df_training,
    classifier,
    n_train,
    train_top_data,
    feature_cols,
    hyper_parameters,
    q_cut_train,
    frac_tp,
    feature_importance,
    clfs,
    i_it,
    score_col,
    training_data,
):
    """
    Fit model_at.
    Args:
        feature_importance:
        df_training (pd.DataFrame): dataframe containing top target and top decoy for each spectrum
        classifier (str): which model to fit
        n_train (int): number of training iterations
        train_top_data (bool): if True only train on top target/decoy per spectrum, if False train on all data
        feature_cols (List): list of features to use when fitting the model
        hyper_parameters (Dict): dictionary containing hyper parameters for the model
        q_cut_train (float):    cut off for q-values below which a target PSM is counted as top-target,
                                when setting top-targets to train on
        frac_tp (float): estimate of fraction of true positives in target dataset
        training_data (pd.DataFrame):   list containing data splits. Contains train split (all targets, 50% decoys)
                                        and test split (all targets, other 50% decoys)
        score_col (str): particular original score
        i_it (int): current iteration cycle
        clfs (Dict): dictionary containing a list of classifiers
    Returns:
        clfs (Dict):    dictionary containing a list of classifiers for each initial_score_col, that are fitted
                        to data at each iteration, with scalers for normalizing data
        df_training (pd.DataFrame): input dataframe with score column added
        feature_importance (Dict): dictionary of dataframe containing feature importance for each initial_score_col
    """

    # Fill missing values
    training_data = replace_missing_data_top_targets(training_data)

    # Fit the model to the data
    # Train is top targets and 50% decoys, test is all targets and other 50% decoys

    train_all = training_data[0].copy(deep=True)
    train = get_train_set(
        training_data[0].copy(deep=True),
        score_col,
        frac_tp=frac_tp,
        train_top_data=train_top_data,
        q_cut=q_cut_train,
    )
    test = training_data[1].copy(deep=True)

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(train.loc[:, feature_cols])
    train.loc[:, feature_cols] = scaler.transform(train.loc[:, feature_cols])
    train_all.loc[:, feature_cols] = scaler.transform(train_all.loc[:, feature_cols])
    test.loc[:, feature_cols] = scaler.transform(test.loc[:, feature_cols])

    # Get the classifier
    clfs_sub = get_training_model(classifier, hyper_parameters)
    clfs_sub.fit(train[feature_cols], train["Is decoy"])

    # Add the feature importance
    if i_it >= n_train:
        if classifier in ["RF", "RF-reg", "ADA", "GBT"]:
            feature_importance_sub = clfs_sub.feature_importances_
        elif classifier in ["SVM", "LR"]:
            feature_importance_sub = clfs_sub.coef_[0]
        elif classifier in ["KNN", "GNB"]:
            feature_importance_sub = [0 for _ in range(len(feature_cols))]
        feature_importance.append(feature_importance_sub)

    df_training.loc[:, "model_score"] = -1000
    df_training.loc[:, "model_score_train"] = -1000
    training_data[1].loc[:, "model_score"] = clfs_sub.score_psm(test[feature_cols])
    training_data[0].loc[:, "model_score_train"] = clfs_sub.score_psm(
        train_all[feature_cols]
    )
    df_training.loc[test.index, "model_score"] = training_data[1]["model_score"]
    df_training.loc[train_all.index, "model_score_train"] = clfs_sub.score_psm(
        train_all[feature_cols]
    )

    clfs.append([clfs_sub, scaler])

    return clfs, df_training, feature_importance


def fit(
    df_training,
    classifier,
    n_train,
    n_eval,
    train_top_data,
    use_cross_validation,
    feature_cols,
    initial_score_col,
    hyper_parameters,
    q_cut,
    q_cut_train,
    frac_tp,
):
    """
    Fit a given model to the data. Training data consists of targets with q-values < 1%
    and 50% of the decoys. Test data is all targets and the other 50% of the decoys.
    Validation data is targets/decoys from a different experiment (if provided).
    Args:
        df_training (pd.DataFrame): dataframe containing top target and top decoy for each spectrum
        classifier (str): which model to fit
        n_train (int): number of training iterations
        n_eval (int): number of of iterations to average score over at the end of the run
        train_top_data (bool): if True only train on top target/decoy per spectrum, if False train on all data
        use_cross_validation (bool): if True use cross validation
        feature_cols (List): list of features to use when fitting the model
        hyper_parameters (Dict): dictionary containing hyper parameters for the model
        initial_score_col (str):    column to use when initially ordering the data (i.e. a search engine score
                                    column). If None, repeat the method using all search engine score
                                    columns one at a time.
        q_cut (float): cut off for q-values below which a target PSM is counted as top-target
        q_cut_train (float):    cut off for q-values below which a target PSM is counted as top-target,
                                when setting top-targets to train on
        frac_tp (float): estimate of fraction of true positives in target dataset
    Returns:
        clfs (Dict):    dictionary containing a list of classifiers for each initial_score_col, that are fitted
                        to data at each iteration, with scalers for normalizing data
        psms (Dict):    dictionary of dictionaries containing number of PSMs identified for train, test and
                        validation data for each initial_score_col
        psms_avg (Dict): like psms but calculated on average score
        psms_engine (Dict): dictionary of dictionaries containing number of PSMs identified by search engine
                            for train, test and validation data, for each initial_score_col
        df (pd.DataFrame): input dataframe with score column added
        feature_importance (Dict): dictionary of dataframe containing feature importance for each initial_score_col
    """
    # [TRISTAN] fit_model, fit_model_at and fit_model_cv in now partly one combined function
    if initial_score_col is None:
        initial_score_cols = [
            score_col for score_col in feature_cols if "Score_processed_" in score_col
        ]
        initial_score_cols = [
            score_col
            for score_col in initial_score_cols
            if all(ml_method not in score_col for ml_method in classifier.ml_methods)
        ]
    else:
        initial_score_cols = [initial_score_col]

    psms_engine = {}
    clfs = []
    feature_importance = []
    psms = {"train": [], "test": []}
    psms_avg = {"train": [], "test": []}
    df_feature_importance = None

    # Train using either method
    for initial_score_col in initial_score_cols:
        print(f"Training from: {initial_score_col}")
        if f"Score_processed_{classifier}" in df_training.columns:
            df_training = df_training.drop(f"Score_processed_{classifier}", axis=1)
        warnings.filterwarnings("ignore", category=DataConversionWarning)

        # Split the training data
        training_data = get_train_test_sets(
            df_training, use_cross_validation=use_cross_validation
        )

        # Get number of PSMs with q-value<1% using search engine score
        if use_cross_validation is True:
            psms_engine["all_data"] = calc_num_psms(
                df_training, initial_score_col, q_cut=q_cut, frac_tp=frac_tp
            )
        else:
            psms_engine["train"] = calc_num_psms(
                training_data[0], initial_score_col, q_cut=q_cut, frac_tp=frac_tp
            )
            psms_engine["test"] = calc_num_psms(
                training_data[1], initial_score_col, q_cut=q_cut, frac_tp=frac_tp
            )

        # Set the total number of iterations
        n_iter = n_train + n_eval

        # Get the original columns
        old_cols = set(df_training.columns)

        # [TRISTAN] necessary?
        for i_it in range(n_iter):
            # Make the training data from targets with q-values < 1% and a fraction of decoys
            if i_it == 0:
                # Rank by original score
                score_col = initial_score_col
                df_training["model_score_all"] = [[]] * len(df_training)
                df_training["model_score_train_all"] = [[]] * len(df_training)
            else:
                # Rank by new score
                score_col = "model_score_train"

            # Use of cross validation depending on set parameter
            if use_cross_validation is True:
                clfs, df_training, feature_importance = fit_model_cv(
                    df_training,
                    classifier,
                    n_train,
                    train_top_data,
                    feature_cols,
                    hyper_parameters,
                    q_cut_train,
                    frac_tp,
                    feature_importance,
                    clfs,
                    i_it,
                    score_col,
                    training_data,
                    old_cols,
                )

            else:
                clfs, df_training, feature_importance = fit_model_at(
                    df_training,
                    classifier,
                    n_train,
                    train_top_data,
                    feature_cols,
                    hyper_parameters,
                    q_cut_train,
                    frac_tp,
                    feature_importance,
                    clfs,
                    i_it,
                    score_col,
                    training_data,
                )

            # start averaging the scores after the initial training iterations are done
            if i_it >= n_train:
                df_training.loc[:, "model_score_all"] += df_training[
                    "model_score"
                ].apply(lambda x: [x])
                df_training.loc[:, "model_score_avg"] = df_training[
                    "model_score_all"
                ].apply(np.mean)
                df_training.loc[:, "model_score_train_all"] += df_training[
                    "model_score_train"
                ].apply(lambda x: [x])
                df_training.loc[:, "model_score_train_avg"] = df_training[
                    "model_score_train_all"
                ].apply(np.mean)

            # see how many PSMs are below 1% in the target set
            psms["test"].append(
                calc_num_psms(df_training, "model_score", q_cut=q_cut, frac_tp=frac_tp)
            )
            psms["train"].append(
                calc_num_psms(
                    df_training, "model_score_train", q_cut=q_cut, frac_tp=frac_tp
                )
            )

            if i_it >= n_train:
                psms_avg["test"].append(
                    calc_num_psms(
                        df_training, "model_score_avg", q_cut=q_cut, frac_tp=frac_tp
                    )
                )
                psms_avg["train"].append(
                    calc_num_psms(
                        df_training,
                        "model_score_train_avg",
                        q_cut=q_cut,
                        frac_tp=frac_tp,
                    )
                )

            if use_cross_validation is True:
                if i_it == 0:
                    print("Iteration\t", "PSMs(train)\t", "PSMs(test)\t")
                print(
                    f"{i_it + 1}\t\t\t",
                    psms["train"][i_it],
                    "\t\t\t",
                    psms["test"][i_it],
                )

            else:
                print(i_it + 1, psms["train"][i_it], psms["test"][i_it])

        sigma = np.std(feature_importance, axis=0)
        feature_importance = np.mean(feature_importance, axis=0)
        df_feature_importance = pd.DataFrame(
            {"feature_importance": feature_importance, "standard deviation": sigma},
            index=feature_cols,
        )
        df_feature_importance = df_feature_importance.sort_values(
            "feature_importance", ascending=False
        )

        df_training = df_training.rename(
            {"model_score_avg": f"Score_processed_{classifier}"}, axis=1
        )
        cols_to_drop = [c for c in df_training.columns if "model_score" in c]
        df_training = df_training.drop(cols_to_drop, axis=1)

    return clfs, psms, psms_avg, psms_engine, df_training, df_feature_importance
