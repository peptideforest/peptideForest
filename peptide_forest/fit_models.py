"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Fit a model to target-decoy data using percolator iterative method
"""

# pylint: disable=unused-import
from typing import Any, Dict, Pattern, Set, Tuple, List
import warnings

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning

from . import models as clf_models
from . import setup_dataset
from .models import ml_methods


def get_q_vals(
    df: pd.DataFrame,
    score_col: str,
    fac: float = 0.9,
    top_psm_only: bool = True,
    get_fdr: bool = True,
    from_method: str = None,
) -> pd.DataFrame:
    """
  Calculate q-value for each PSM based on the score in the column score_col
  Arguments:
    - df: dataframe containing experimental data
    - score_col: name of column containing score to rank PSMs by
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    - top_psm_only: keep only highest scoring PSM for each spectrum, default=True
    - get_fdr: return the false detection rate as q-value (default), else return q-values
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - df_scores: dataframe containing q-values, score and whether PSM is target or decoy
  """
    engines = [c for c in df.columns if "Score_processed" in c]
    engines = [
        e for e in engines if e not in [f"Score_processed_{m}" for m in ml_methods]
    ]
    engines = [e.split("Score_processed_")[-1] for e in engines]

    not_in_engines = any([e in score_col for e in engines])

    if from_method and not not_in_engines:
        df_scores = df.sort_values(
            [score_col, f"Score_processed_{from_method}"], ascending=[False, False]
        ).copy(deep=True)
    else:
        df_scores = df.sort_values(score_col, ascending=False).copy(deep=True)

    df_scores = find_psms_to_keep(df_scores, score_col)
    df_scores = df_scores[df_scores["keep in"]]

    if top_psm_only:
        df_scores = df_scores.drop_duplicates("Spectrum ID")

    fac = fac * sum(~df_scores["Is decoy"]) / sum(df_scores["Is decoy"])
    df_scores = df_scores[["Spectrum ID", "Sequence", score_col, "Is decoy"]]
    target_decoy_dummies = pd.get_dummies(df_scores["Is decoy"])
    df_scores.loc[
        :, str(target_decoy_dummies.columns[0])
    ] = target_decoy_dummies.values[:, 0]
    df_scores.loc[
        :, str(target_decoy_dummies.columns[1])
    ] = target_decoy_dummies.values[:, 1]
    df_scores = df_scores.rename({"False": "Target", "True": "Decoy"}, axis=1)
    df_scores["FDR"] = fac * df_scores["Decoy"].cumsum() / df_scores["Target"].cumsum()
    if get_fdr:
        df_scores["q-value"] = df_scores["FDR"]
    else:
        df_scores["q-value"] = df_scores["FDR"].cummax()
    return df_scores


def find_psms_to_keep(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
  Flag psms to remove from q-value calculations, when the highest score for a given
  spectrum is shared by two or more PSMs.
    - if the highest scoring PSMs are both decoys, pick one at random to keep in
    - if the highest scoring PSMs are both targets, flag to remove entire spectrum
    - if at least one is a decoy and at least one is a target, flag all to remove entire spectrum
  Arguments:
    - df: dataframe containing search engine scores for all psms
    - score_col: score column to use when comparing PSM scores (e.g. search engine score
      or score from an ML method)
  Returns:
    - df: updated dataframe with flag 'keep in': False means remove when calculating
          q-values
  """
    max_col = "max_" + score_col.split("Score_processed_")[-1]
    eq_col = "eq_" + score_col.split("Score_processed_")[-1]

    # get the maximum value for each spectrum ID
    df.loc[:, max_col] = df.groupby("Spectrum ID")[score_col].transform("max")
    # replace 0 with nan, as when the maximum score is 0, there are no PSMs
    df.loc[:, max_col] = df[max_col].replace(0.0, np.nan)
    df.loc[:, eq_col] = df[max_col] == df[score_col]

    # flag to keep psm in set
    df["keep in"] = True

    # number psms with max score for each spectrum
    num_with_top_val = df.groupby("Spectrum ID")[eq_col].sum()
    # index of spectra where more than one psm has the max score
    inds = num_with_top_val.index[num_with_top_val > 1]
    # get dataframe for those spectra and take only psms that have the max score
    df_with_top_val = df[df["Spectrum ID"].isin(inds) & df[eq_col]].copy(deep=True)
    # check how many decoys there are for each spectra
    df_num_decoys = df_with_top_val.groupby("Spectrum ID")["Is decoy"].agg(
        ["count", "sum", "nunique"]
    )
    # get spectra where all PSMs with max_score should be dropped
    # target and decoy are ranked top
    inds_drop = set(df_num_decoys[df_num_decoys["nunique"] == 2].index)
    # only targets are top
    inds_drop = inds_drop.union(set(df_num_decoys[df_num_decoys["sum"] == 0].index))

    # get spectra where there are only decoys with the max score, one of which will be kept
    inds_keep = df_num_decoys[df_num_decoys["sum"] == df_num_decoys["count"]].index

    # get list of PSMs to drop: choose one at random to keep when both were decoys
    psms_keep = list(
        df[df["Spectrum ID"].isin(inds_keep) & df[eq_col]]
        .sample(frac=1.0)
        .drop_duplicates("Spectrum ID")
        .index
    )

    # and drop the other one
    psms_drop = list(
        df[
            df["Spectrum ID"].isin(inds_keep) & df[eq_col] & (~df.index.isin(psms_keep))
        ].index
    )

    # add those that where all PSMs with max_score should be dropped
    # drop all PSMs for those spectrums
    psms_drop += list(df[df["Spectrum ID"].isin(inds_drop)].index)

    # change to False if in psms_drop
    df.loc[df.index.isin(psms_drop), "keep in"] = False
    # remove columns that are no longer needed
    df = df.drop([max_col, eq_col], axis=1)
    return df


def get_top_targets(
    df: pd.DataFrame, score_col: str, q_cut: float = 0.01, fac: float = 0.9
) -> pd.DataFrame:
    """
  Return only target PSMs with q-value less than 1%
  Arguments:
    - df: dataframe containing experimental data
    - score_col: name of column containing score to rank PSMs by
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - df.loc[inds, :]: inds is the index of PSMs matchine criteria
  """
    df_scores = get_q_vals(df, score_col, fac=fac, top_psm_only=False, get_fdr=False)
    inds = df_scores[(df_scores["q-value"] <= q_cut) & (~df_scores["Is decoy"])].index
    return df.loc[inds, :]


def get_train_set(
    df: pd.DataFrame,
    score_col: str,
    q_cut: float = 0.01,
    fac: float = 0.9,
    train_top_data: bool = False,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
  Return training dataset containing sample of decoys and all target PSMs with q-value less than 1%
  Arguments:
    - df: dataframe containing experimental data
    - score_col: name of column containing score to rank PSMs by
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    - train_top_data: if False (default), train on all the data. If True, only train on top
      target/decoy per spectrum
    - sample_frac: ratio of decoy PSMs to target PSMs in the training dataset
  Returns:
    - train: dataframe containing selected target and decoy PSMs
  """
    if train_top_data:
        # take only the top target and top decoy for each spectrum
        df_to_train = setup_dataset.get_top_target_decoy(df, score_col=score_col)
    else:
        df_to_train = df.copy(deep=True)
    train_targets = get_top_targets(df_to_train, score_col, q_cut=q_cut, fac=fac)
    n_sample = int(sample_frac * len(train_targets))
    if n_sample > len(df_to_train[df_to_train["Is decoy"]]):
        train_decoys = df_to_train[df_to_train["Is decoy"]].sample(
            n=n_sample, replace=True
        )
    else:
        train_decoys = df_to_train[df_to_train["Is decoy"]].sample(n=n_sample)
    # concatonate and reorder
    train = pd.concat([train_targets, train_decoys]).sample(frac=1)
    return train


def get_train_test_sets(df: pd.DataFrame, use_cv: bool = True) -> List:
    """
  Split dataframe into multiple sets for training
  Arguments:
    - df: dataframe containing experimental data
    - use_cv: if True (default), use 3-fold cross-validation. If False, trian on all targets
  Return:
    - training_data: list containing data splits. If use_cv is True, then has 3 cv splits. If
                     use_cv is False, contains train split (all targets, 50% decoys) and test
                     split (all targets, other 50% decoys)
  """
    # start a list to contain the data
    training_data = []
    # if use_cv, split data into 3 equal sets of spectrum IDs
    if use_cv:
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


def get_scores(
    scores_in: Dict,
    clf: Any,
    norm_data: pd.DataFrame,
    data: pd.DataFrame,
    split_name: str,
) -> Dict:
    """
  Get the accuracy for targets, decoys, and for both
  Arguments:
    - scores_in: dictionary containing scores for each of the above
    - clf: classifier to make predictions with
    - norm_data: normalized data for the features for each PSM
    - data: data for each PSM
    - split_name: name of split (train, test or val) that data is for
  Returns:
    - scores_in: dictionary with new scores added
  """
    decoys = data.loc[data.loc[:, "Is decoy"], "Is decoy"]
    targets = data.loc[~data.loc[:, "Is decoy"], "Is decoy"]
    decoys_s = norm_data[data["Is decoy"]]
    targets_s = norm_data[~data["Is decoy"]]
    score_d = clf.score(decoys_s, decoys)
    score_t = clf.score(targets_s, targets)
    score = clf.score(norm_data, data["Is decoy"])
    scores_in[split_name]["all"].append(score)
    scores_in[split_name]["targets"].append(score_t)
    scores_in[split_name]["decoys"].append(score_d)
    return scores_in


def calc_num_psms(
    df: pd.DataFrame, score_col: str, q_cut: float = 0.01, fac: float = 0.9
) -> int:
    """
  Calculate the number of PSMs with q-value < 1%, using the following criteria
    - only the highest ranked PSMs for each spectra are considered
    - redundent (i.e. duplicate) peptide sequences are discarded
    - only target PSMs are counted
  Arguments:
    - df: dataframe containing scores for each PSM
    - score_col: name of column to rank PSMs by
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
  Returns:
    - n_psms: number of PSMs with q-value < 1% that match the above criteria
  """
    # get the q-values
    df_scores_new = get_q_vals(df, score_col, fac=fac, top_psm_only=True)
    # order by the q-value, and keep only the first PSM for each spectra
    df_scores_sub = df_scores_new.sort_values("q-value", ascending=True)
    df_scores_sub = df_scores_sub.drop_duplicates("Spectrum ID")
    # get the number of target PSMs with q-value < 1%
    n_psms = len(
        df_scores_sub.loc[
            (df_scores_sub["q-value"] <= q_cut) & (~df_scores_sub["Is decoy"]), :
        ]
    )
    return n_psms


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def fit_model_cv(
    model_type: str,
    features: List,
    df: pd.DataFrame,
    n_train: int = 10,
    n_eval: int = 15,
    hp_dict_in: Dict = None,
    initial_score_col: str = "Score_processed",
    verbose: int = 2,
    train_top_data: bool = False,
    q_cut: float = 0.01,
    q_cut_train: float = 0.01,
    fac: float = 0.9,
) -> Tuple[Any, Dict, Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
  Fit a given model to the data. Training data consists of targets with q-values < 1%
  and 50% of the decoys. Test data is all targets and the other 50% of the decoys.
  Validation data is targets/decoys from a different experiment (if provided)
  Arguments:
    - model_type: which model to fit (SVM or RF)
    - features: list of features to use when fitting the model
    - df: dataframe containing top target and top decoy for each spectrum
    - n_train: number of training iterations, default = 10
    - n_eval: number of iterations to average score over at the end of the run
    - hp_dict_in: dictionary containing hyperparameters for the model
    - verbose: what to print. 1 = final result, 2 = results at each iteration. Otherwise
               print nothing. Default = 2.
    - train_top_data: if False (default), train on all the data. If True, only train on top
      target/decoy per spectrum
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - q_cut_train: cut off for q-values, below which a target PSM is counted as a top-target, when
                   setting top-targets to train on. Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    Returns:
    - clfs: list of classifiers that are fitted to data at each iteration, with scalers
    - psms: dictionary containing number of PSMs identified for train, test and validation data
    - psms_avg: same as above but calculated on average score
    - psms_engine: dictionary containing number of PSMs identified by search engine for train,
                   test and validation data
    - df: input dataframe with score column added
    - df_feature_importance: dataframe containing feature importance
  """
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    clfs = []  # type: List[Any]
    psms_engine = {}
    psms = {"train": [], "test": []}  # type: Dict[str, List]
    psms_avg = {"train": [], "test": []}  # type: Dict[str, List]

    # split the training data
    training_data = get_train_test_sets(df, use_cv=True)

    # get number of PSMs with q-value<1% using search engine score
    psms_engine["all_data"] = calc_num_psms(df, initial_score_col, q_cut=q_cut, fac=fac)

    # set the total number of iterations
    n_iter = n_train + n_eval

    # get the original columns
    org_cols = set(df.columns)

    # initialize feature importance
    feature_importance = []

    for i_it in range(n_iter):

        # make the training data from targets with q-values < 1% and a fraction of decoys
        if i_it == 0:
            # rank by original score
            score_col = initial_score_col
            df["model_score_all"] = [[]] * len(df)
            df["model_score_train_all"] = [[]] * len(df)
        else:
            # rank by new score
            score_col = "model_score_train"

        # fit the model to the data

        clfs_sub = []
        df.loc[:, "model_score_train"] = [0 for _ in np.arange(len(df))]
        for it in np.arange(len(training_data)):
            j, k = [s for s in [0, 1, 2] if s != it]
            # copy the data (so that the scaling of the data does not change the original)
            test = training_data[it].copy(deep=True)
            train_j = training_data[j][list(org_cols.union([score_col]))].copy(
                deep=True
            )
            train_k = training_data[k][list(org_cols.union([score_col]))].copy(
                deep=True
            )

            # fill in missing values
            train_j, train_k, test = setup_dataset.replace_missing_data_cv(
                train_j, train_k, test
            )

            # train on 2/3 data, with top targets and 50% decoys
            train = pd.concat([train_j, train_k])
            train = get_train_set(
                train,
                score_col,
                fac=0.9,
                train_top_data=train_top_data,
                q_cut=q_cut_train,
            )

            # scale the data
            scaler = preprocessing.StandardScaler().fit(train.loc[:, features])
            train.loc[:, features] = scaler.transform(train.loc[:, features])
            train_j.loc[:, features] = scaler.transform(train_j.loc[:, features])
            train_k.loc[:, features] = scaler.transform(train_k.loc[:, features])
            test.loc[:, features] = scaler.transform(test.loc[:, features])
            # get the classifier
            clf = clf_models.get_training_model(model_type, hp_dict_in)
            clf.fit(train[features], train["Is decoy"])
            clfs_sub.append([clf, scaler])

            # add the feature importance
            if i_it >= n_train:
                if model_type.upper() in ["RF", "RF-REG", "ADA", "GBT"]:
                    feature_importance_sub = clf.feature_importances_
                elif model_type.upper() in ["SVM", "LR"]:
                    feature_importance_sub = clf.coef_[0]
                elif model_type.upper() in ["KNN", "GNB"]:
                    feature_importance_sub = [0 for _ in range(len(features))]
                feature_importance.append(feature_importance_sub)

            # score each of the splits
            # pylint: disable=not-callable
            training_data[it].loc[:, "model_score"] = clf.score_psm(test[features])
            training_data[j].loc[:, "model_score_train"] = clf.score_psm(
                train_j[features]
            )
            training_data[k].loc[:, "model_score_train"] = clf.score_psm(
                train_k[features]
            )
            df.loc[training_data[it].index, "model_score"] = training_data[it][
                "model_score"
            ]
            # get the training score, which we take as the average score over each time the
            # psm is trained on
            df.loc[train_j.index, "model_score_train"] += training_data[j][
                "model_score_train"
            ].values
            df.loc[train_k.index, "model_score_train"] += training_data[k][
                "model_score_train"
            ].values
        df.loc[:, "model_score_train"] = df["model_score_train"] / (
            len(training_data) - 1
        )
        clfs.append(clfs_sub)
        # start averaging the scores after the initial training iterations are done
        if i_it >= n_train:
            df.loc[:, "model_score_all"] += df["model_score"].apply(lambda x: [x])
            df.loc[:, "model_score_avg"] = df["model_score_all"].apply(np.mean)
            df.loc[:, "model_score_train_all"] += df["model_score_train"].apply(
                lambda x: [x]
            )
            df.loc[:, "model_score_train_avg"] = df["model_score_train_all"].apply(
                np.mean
            )

        # see how many PSMs are below 1% in the test and train set
        psms["test"].append(calc_num_psms(df, "model_score", q_cut=q_cut, fac=fac))
        psms["train"].append(
            calc_num_psms(df, "model_score_train", q_cut=q_cut, fac=fac)
        )

        if i_it >= n_train:
            psms_avg["test"].append(
                calc_num_psms(df, "model_score_avg", q_cut=q_cut, fac=fac)
            )
            psms_avg["train"].append(
                calc_num_psms(df, "model_score_train_avg", q_cut=q_cut, fac=fac)
            )

        if verbose == 2:
            print(
                f"Iteration {i_it + 1: >3}",
                i_it + 1,
                psms["train"][i_it],
                psms["test"][i_it],
            )
    if verbose in [1, 2]:
        print(psms_avg["train"][-1], psms_avg["test"][-1], "\n")

    sigma = np.std(feature_importance, axis=0)
    feature_importance = np.mean(feature_importance, axis=0)
    df_feature_importance = pd.DataFrame(
        {"feature_importance": feature_importance, "standard deviation": sigma},
        index=features,
    )
    df_feature_importance = df_feature_importance.sort_values(
        "feature_importance", ascending=False
    )

    df = df.rename({"model_score_avg": f"Score_processed_{model_type}"}, axis=1)
    cols_to_drop = [c for c in df.columns if "model_score" in c]
    df = df.drop(cols_to_drop, axis=1)
    return clfs, psms, psms_avg, psms_engine, df, df_feature_importance


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def fit_model_at(
    model_type: str,
    features: List,
    df: pd.DataFrame,
    n_train: int = 10,
    n_eval: int = 15,
    hp_dict_in: Dict = None,
    initial_score_col: str = "Score_processed",
    verbose: int = 2,
    train_top_data: bool = False,
    q_cut: float = 0.01,
    q_cut_train: float = 0.01,
    fac: float = 0.9,
) -> Tuple[Any, Dict, Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
  Fit a given model to the data. Training data consists of targets with q-values < 1%
  and 50% of the decoys. Test data is all targets and the other 50% of the decoys.
  Validation data is targets/decoys from a different experiment (if provided)
  Arguments:
    - model_type: which model to fit (SVM or RF)
    - features: list of features to use when fitting the model
    - df: dataframe containing top target and top decoy for each spectrum
    - n_train: number of training iterations, default = 10
    - n_eval: number of iterations to average score over at the end of the run
    - hp_dict_in: dictionary containing hyperparameters for the model
    - verbose: what to print. 1 = final result, 2 = results at each iteration. Otherwise
               print nothing. Default = 2.
    - train_top_data: if False (default), train on all the data. If True, only train on top
      target/decoy per spectrum
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - q_cut_train: cut off for q-values, below which a target PSM is counted as a top-target, when
                   setting top-targets to train on. Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    Returns:
    - clfs: list of classifiers that are fitted to data at each iteration, with scalers
    - psms: dictionary containing number of PSMs identified for train, test and validation data
    - psms_avg: same as above but calculated on average score
    - psms_engine: dictionary containing number of PSMs identified by search engine for train,
                   test and validation data
    - df: input dataframe with score column added
    - df_feature_importance: dataframe containing feature importance
  """
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    clfs = []  # type: List[Any]
    psms_engine = {}
    psms = {"train": [], "test": []}  # type: Dict[str, List]
    psms_avg = {"train": [], "test": []}  # type: Dict[str, List]

    # split the training data
    training_data = get_train_test_sets(df, use_cv=False)

    # get number of PSMs with q-value<1% using search engine score
    psms_engine["train"] = calc_num_psms(
        training_data[0], initial_score_col, q_cut=q_cut, fac=fac
    )
    psms_engine["test"] = calc_num_psms(
        training_data[1], initial_score_col, q_cut=q_cut, fac=fac
    )

    # set the total number of iterations
    n_iter = n_train + n_eval

    # initialize feature importance
    feature_importance = []

    for i_it in range(n_iter):

        # fill in missing values
        training_data = setup_dataset.replace_missing_data_all_targets(training_data)

        # make the training data from targets with q-values < 1% and a fraction of decoys
        if i_it == 0:
            # rank by original score
            score_col = initial_score_col
            df["model_score_all"] = [[]] * len(df)
            df["model_score_train_all"] = [[]] * len(df)
        else:
            # rank by new score
            score_col = "model_score_train"

        # fit the model to the data
        # train is top targets and 50% decoys, test is all targets and other 50% decoys

        train_all = training_data[0].copy(deep=True)
        train = get_train_set(
            training_data[0].copy(deep=True),
            score_col,
            fac=0.9,
            train_top_data=train_top_data,
            q_cut=q_cut_train,
        )
        test = training_data[1].copy(deep=True)
        # scale the data
        scaler = preprocessing.StandardScaler().fit(train.loc[:, features])
        train.loc[:, features] = scaler.transform(train.loc[:, features])
        train_all.loc[:, features] = scaler.transform(train_all.loc[:, features])
        test.loc[:, features] = scaler.transform(test.loc[:, features])
        # get the classifier
        clfs_sub = clf_models.get_training_model(model_type, hp_dict_in)
        clfs_sub.fit(train[features], train["Is decoy"])

        # add the feature importance
        if i_it >= n_train:
            if model_type in ["RF", "RF-reg", "ADA", "GBT"]:
                feature_importance_sub = clfs_sub.feature_importances_
            elif model_type in ["SVM", "LR"]:
                feature_importance_sub = clfs_sub.coef_[0]
            elif model_type in ["KNN", "GNB"]:
                feature_importance_sub = [0 for _ in range(len(features))]
            feature_importance.append(feature_importance_sub)

        df.loc[:, "model_score"] = -1000
        df.loc[:, "model_score_train"] = -1000
        # pylint: disable=not-callable
        training_data[1].loc[:, "model_score"] = clfs_sub.score_psm(test[features])
        # pylint: disable=not-callable
        training_data[0].loc[:, "model_score_train"] = clfs_sub.score_psm(
            train_all[features]
        )
        df.loc[test.index, "model_score"] = training_data[1]["model_score"]
        # pylint: disable=not-callable
        df.loc[train_all.index, "model_score_train"] = clfs_sub.score_psm(
            train_all[features]
        )

        clfs.append([clfs_sub, scaler])
        # start averaging the scores after the initial training iterations are done
        if i_it >= n_train:
            df.loc[:, "model_score_all"] += df["model_score"].apply(lambda x: [x])
            df.loc[:, "model_score_avg"] = df["model_score_all"].apply(np.mean)
            df.loc[:, "model_score_train_all"] += df["model_score_train"].apply(
                lambda x: [x]
            )
            df.loc[:, "model_score_train_avg"] = df["model_score_train_all"].apply(
                np.mean
            )

        # see how many PSMs are below 1% in the target set
        psms["test"].append(calc_num_psms(df, "model_score", q_cut=q_cut, fac=fac))
        psms["train"].append(
            calc_num_psms(df, "model_score_train", q_cut=q_cut, fac=fac)
        )

        if i_it >= n_train:
            psms_avg["test"].append(
                calc_num_psms(df, "model_score_avg", q_cut=q_cut, fac=fac)
            )
            psms_avg["train"].append(
                calc_num_psms(df, "model_score_train_avg", q_cut=q_cut, fac=fac)
            )

        if verbose == 2:
            print(i_it + 1, psms["train"][i_it], psms["test"][i_it])
    if verbose in [1, 2]:
        print(psms_avg["train"][-1], psms_avg["test"][-1], "\n")

    sigma = np.std(feature_importance, axis=0)
    feature_importance = np.mean(feature_importance, axis=0)
    df_feature_importance = pd.DataFrame(
        {"feature_importance": feature_importance, "standard deviation": sigma},
        index=features,
    )
    df_feature_importance = df_feature_importance.sort_values(
        "feature_importance", ascending=False
    )

    df = df.rename({"model_score_avg": f"Score_processed_{model_type}"}, axis=1)
    cols_to_drop = [c for c in df.columns if "model_score" in c]
    df = df.drop(cols_to_drop, axis=1)
    return clfs, psms, psms_avg, psms_engine, df, df_feature_importance


def score_from_model(
    df: pd.DataFrame, features: List, classifiers: List
) -> pd.DataFrame:
    """
  Score psms based on a previously fitted model, at each iteration the model was fitted.
  The fitting type is inferred (either the original percolator method, or 3-fold cross-validation)
    - If 3-fold cv method was used, predictions are made using the average of the
      scores from 3 models
    - If the original method was used, only the one model is available at each iteration
  Arguments:
    - df: dataframe containing features for each psm
    - features: list of features used for the models
    - classifiers: list containing the fitted model and standard scaler at each iteration
  Returns:
    - df: same dataframe, with scores from fitted model
  """
    if len(classifiers[0]) == 3:
        for i, classifiers_sub in enumerate(classifiers):
            col = f"score_processed_fitted_model_iter_{i+1}"
            # initialize scores to zero for this iteration
            df[col] = np.zeros(len(df))
            for classifier, scaler in classifiers_sub:
                # get the scaled feature values
                data = scaler.transform(df[features])
                # get the score and add to the sum
                df.loc[:, col] += classifier.score_psm(data)
            # take the average score
            df.loc[:, col] = df[col] / len(classifiers_sub)
    elif len(classifiers[0]) == 2:
        for i, (classifier, scaler) in enumerate(classifiers):
            col = f"score_processed_fitted_model_iter_{i+1}"
            # get the scaled feature values
            data = scaler.transform(df[features])
            # get the score
            df[col] = classifier.score_psm(data)
    else:
        # something wrong with the data format
        print("Error with classifiers: incorrect data format")
    return df


# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-arguments
def fit_model(
    model_type: str,
    features: List,
    df: pd.DataFrame,
    n_train: int = 10,
    n_eval: int = 15,
    hp_dict_in: Dict = None,
    initial_score_col: str = None,
    verbose: int = 2,
    train_top_data: bool = False,
    use_cv: bool = True,
    q_cut: float = 0.01,
    q_cut_train: float = 0.01,
    fac: float = 0.9,
) -> Tuple[Any, Dict, Dict, Dict, pd.DataFrame, pd.DataFrame]:
    """
  Fit a given model to the data. Training data consists of targets with q-values < 1%
  and 50% of the decoys. Test data is all targets and the other 50% of the decoys.
  Validation data is targets/decoys from a different experiment (if provided)
  Arguments:
    - model_type: which model to fit (SVM or RF)
    - features: list of features to use when fitting the model
    - df: dataframe containing top target and top decoy for each spectrum
    - n_train: number of training iterations, default = 10
    - n_eval: number of iterations to average score over at the end of the run
    - hp_dict_in: dictionary containing hyperparameters for the model
    - initial_score_col: column to use when initially ordering the data (i.e. a search engine score
                         column). If None (default), repeat the method using all search engine score
                         columns one at a time.
    - verbose: what to print. 1 = final result, 2 = results at each iteration. Otherwise
               print nothing. Default = 2.
    - train_top_data: if False (default), train on all the data. If True, only train on top
      target/decoy per spectrum
    - use_cv: if True (default), use 3-fold cross-validation. If False, trian on all targets
    - q_cut: cut off for q-values, below which a target PSM is counted as a top-target.
             Default is 0.01
    - q_cut_train: cut off for q-values, below which a target PSM is counted as a top-target, when
                   setting top-targets to train on. Default is 0.01
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    Returns:
    - clfs: dictionary containing a list of classifiers for each initial_score_col, that are fitted
            to data at each iteration, with scalers for normalizing data
    - psms: dictionary of dictionaries containing number of PSMs identified for train, test and
            validation data for each initial_score_col
    - psms_avg: same as above but calculated on average score
    - psms_engine: dictionary of dictionaries containing number of PSMs identified by search engine
                   for train, test and validation data, for each initial_score_col
    - df: input dataframe with score column added
    - feature_importances: dictionary of dataframe containing feature importance for each
                           initial_score_col
  """

    if initial_score_col is None:
        initial_score_cols = [
            score_col for score_col in features if "Score_processed_" in score_col
        ]
        initial_score_cols = [
            score_col
            for score_col in initial_score_cols
            if all(ml_method not in score_col for ml_method in clf_models.ml_methods)
        ]
    else:
        initial_score_cols = [initial_score_col]

    clfs = {}
    psms = {}
    psms_avg = {}
    psms_engine = {}
    feature_importances = {}

    # train using either method
    for initial_score_col in initial_score_cols:
        print(f"Training from: {initial_score_col}")
        if f"Score_processed_{model_type}" in df.columns:
            df = df.drop(f"Score_processed_{model_type}", axis=1)
        if use_cv:
            (
                clfs[initial_score_col],
                psms[initial_score_col],
                psms_avg[initial_score_col],
                psms_engine[initial_score_col],
                df,
                feature_importances[initial_score_col],
            ) = fit_model_cv(
                model_type,
                features,
                df,
                n_train=n_train,
                n_eval=n_eval,
                hp_dict_in=hp_dict_in,
                initial_score_col=initial_score_col,
                verbose=verbose,
                train_top_data=train_top_data,
                q_cut=q_cut,
                q_cut_train=q_cut_train,
                fac=fac,
            )
        else:
            (
                clfs[initial_score_col],
                psms[initial_score_col],
                psms_avg[initial_score_col],
                psms_engine[initial_score_col],
                df,
                feature_importances[initial_score_col],
            ) = fit_model_at(
                model_type,
                features,
                df,
                n_train=n_train,
                n_eval=n_eval,
                hp_dict_in=hp_dict_in,
                initial_score_col=initial_score_col,
                verbose=verbose,
                train_top_data=train_top_data,
                q_cut=q_cut,
                q_cut_train=q_cut_train,
                fac=fac,
            )
        df[f"score_{model_type}_from_{initial_score_col}".lower()] = df[
            f"Score_processed_{model_type}"
        ]

    df = df.drop(f"Score_processed_{model_type}", axis=1)
    cols = [
        f"score_{model_type}_from_{initial_score_col}".lower()
        for initial_score_col in initial_score_cols
    ]
    df[f"Score_processed_{model_type}"] = df[cols].mean(axis=1)

    return clfs, psms, psms_avg, psms_engine, df, feature_importances
