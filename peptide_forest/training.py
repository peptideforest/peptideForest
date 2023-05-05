"""Train peptide forest."""
import multiprocessing as mp
import sys
import types

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from peptide_forest import knowledge_base


def find_psms_to_keep(df_scores, score_col):
    """Remove PSMs from q-value calculations if the highest score for a given spectrum is shared by two or more PSMs.

    Args:
        df_scores (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): column to score PSMs by

    Returns:
        df_scores (pd.DataFrame): updated dataframe
    """
    # Get maxima per spectrum ID
    max_per_spec_id = (
        df_scores.groupby("spectrum_id")[score_col].transform(max).replace(0.0, pd.NA)
    )
    # Find those PSMs where the score equals the maximum for that spectrum
    score_is_max = max_per_spec_id == df_scores[score_col]
    # Find Spectrum IDs where condition is met by one PSM
    specs_with_more_than_one_top_score = (
        df_scores.loc[score_is_max].groupby("spectrum_id")[score_col].count() > 1
    )
    specs_with_more_than_one_top_score = specs_with_more_than_one_top_score[
        specs_with_more_than_one_top_score
    ].index
    # Get stats for decoys in those spectrums
    decoys_per_spec = (
        df_scores[
            df_scores["spectrum_id"].isin(specs_with_more_than_one_top_score)
            & score_is_max
        ]
        .groupby("spectrum_id")["is_decoy"]
        .agg("sum")
    )

    # Mark spectra which are to be removed
    # Condition is remove spectra with more than one top scoring PSM, if at least one of the top PSMs is a decoy

    spec_ids_drop = decoys_per_spec > 0
    spec_ids_drop = spec_ids_drop[spec_ids_drop].index

    # Drop them
    drop_inds = df_scores["spectrum_id"].isin(spec_ids_drop)
    df_scores = df_scores[~drop_inds]

    return df_scores


def calc_q_vals(
    df,
    score_col,
    sensitivity,
    top_psm_only,
    get_fdr,
    init_score_col,
):
    """Calculate q-values for a given scoring column.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): column to score PSMs by
        sensitivity (float): proportion of positive results to true positives in the data
        top_psm_only (bool): drop all but the top scoring PSM (target+decoy) per spectrum
        get_fdr (bool): give q-values as fdr or its cummax
        init_score_col (str): initial engine to rank results by

    Returns:
        df_scores (pd.DataFrames): dataframe with computed q values
    """
    # Create copy to avoid manipulation of original data
    df_scores = df.copy(deep=True)

    # Sort by score_col
    if init_score_col is not None and score_col == "score_processed_peptide_forest":
        df_scores.sort_values(
            [score_col, f"score_processed_{init_score_col}"],
            ascending=[False, False],
            inplace=True,
        )
    else:
        df_scores.sort_values(score_col, ascending=False, inplace=True)

    # Remove PSMs from q-value calculations if the highest score for a given spectrum is shared by two or more PSMs.
    df_scores = find_psms_to_keep(df_scores, score_col)

    if top_psm_only is True:
        # Use only the top PSM
        df_scores = df_scores.drop_duplicates("spectrum_id")

    # Calculate FDR by considering remaining decoys and targets
    df_scores = df_scores[["spectrum_id", "sequence", score_col, "is_decoy"]]
    target_decoy_hot_one = pd.get_dummies(df_scores["is_decoy"]).rename(
        {False: "target", True: "decoy"}, axis=1
    )
    if "decoy" not in target_decoy_hot_one.columns:
        target_decoy_hot_one["decoy"] = 0
    df_scores = pd.concat([df_scores, target_decoy_hot_one], axis=1)
    df_scores["fdr"] = (
        sensitivity
        * df_scores["decoy"].cumsum()
        / (df_scores["target"].cumsum() + df_scores["decoy"].cumsum())
    )

    # Compute q-values
    if get_fdr is True:
        df_scores["q-value"] = df_scores["fdr"]
    else:
        df_scores["q-value"] = df_scores["fdr"].cummax()

    return df_scores


def calc_num_psms(df, score_col, q_cut, sensitivity):
    """Compute number of PSMs for dataframe which meet cutoff criterium.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): initial engine to rank results by
        q_cut (float): q-value cutoff dfor PSM selection
        sensitivity (float): proportion of positive results to true positives in the data

    Returns:
        n_psms (int): number of target PSMs
    """
    # Get the q-values
    df_scores_new = calc_q_vals(
        df,
        score_col,
        sensitivity=sensitivity,
        top_psm_only=True,
        get_fdr=True,
        init_score_col=None,
    )

    # Order by the q-value, and keep only the first PSM for each spectra
    df_scores_sub = df_scores_new.sort_values("q-value", ascending=True)

    # Get the number of target PSMs with q-value < 1%
    n_psms = len(
        df_scores_sub.loc[
            (df_scores_sub["q-value"] <= q_cut) & (~df_scores_sub["is_decoy"]),
            :,
        ]
    )
    return n_psms


def _score_psms(clf, data):
    """Apply scoring function to classifier prediction.

    Args:
        clf (sklearn.ensemble.RandomForestRegressor): trained classifier
        data (array): input data for prediction

    Returns:
        data (array): predictions with applied scoring function
    """
    return 2 * (0.5 - clf.predict(data))


def get_rf_reg_classifier(hyperparameters):
    """Initialize random forest regressor.

    Args:
        hyperparameters (dict): sklearn hyperparameters for classifier

    Returns:
        clf (sklearn.ensemble.RandomForestRegressor): classifier with added method to score PSMs
    """
    clf = RandomForestRegressor(**hyperparameters)
    # Add scoring function
    clf.score_psms = types.MethodType(_score_psms, clf)

    return clf


def fit_cv(df, score_col, cv_split_data, sensitivity, q_cut):
    """Process single-epoch of cross validated training.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): column to score PSMs by
        cv_split_data (list): list with indices of data to split by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits in epoch
    """
    # Reset scores
    df.loc[:, "model_score"] = 0
    df.loc[:, "model_score_train"] = 0

    feature_importances = []

    for i, split in enumerate(cv_split_data):
        train_inds, test_inds = split
        # Create training data copy for current split
        train = df.loc[train_inds.tolist(), :].copy(deep=True)
        test = df.loc[test_inds.tolist(), :].copy(deep=True)

        # Fill missing data
        delta_cols = [c for c in train.columns if "delta_score" in c]
        min_per_delta_col = train[delta_cols].min()
        train.loc[:, delta_cols].fillna(min_per_delta_col, inplace=True)
        test.loc[:, delta_cols].fillna(min_per_delta_col, inplace=True)

        # Use only top target and top decoy per spectrum
        train_data = (
            train.sort_values(score_col, ascending=False)
            .drop_duplicates(["spectrum_id", "is_decoy"])
            .copy(deep=True)
        )

        # Filter for target PSMs with q-value < q_cut
        train_q_vals = calc_q_vals(
            df=train_data,
            score_col=score_col,
            sensitivity=sensitivity,
            top_psm_only=False,
            get_fdr=False,
            init_score_col=None,
        )[["q-value", "is_decoy"]]
        train_q_cut_met_targets = train_q_vals.loc[
            (train_q_vals["q-value"] <= q_cut) & (~train_q_vals["is_decoy"])
        ].index
        train_targets = train_data.loc[train_q_cut_met_targets, :]
        # Get same number of decoys to match targets at random
        train_decoys = train_data[train_data["is_decoy"]].sample(n=len(train_targets))

        # Combine to form training dataset
        train_data = pd.concat([train_targets, train_decoys]).sample(frac=1)

        # Scale the data
        features = list(
            set(train_data.columns).difference(
                set(
                    [
                        c
                        for c in train_data.columns
                        for r in knowledge_base.parameters["non_trainable_columns"]
                        if c.startswith(r)
                    ]
                )
            )
        )
        scaler = StandardScaler().fit(train_data.loc[:, features])
        train_data.loc[:, features] = scaler.transform(train_data.loc[:, features])
        train.loc[:, features] = scaler.transform(train.loc[:, features])
        test.loc[:, features] = scaler.transform(test.loc[:, features])

        # Get RF-reg classifier and train
        hyperparameters = knowledge_base.parameters["hyperparameters"]
        hyperparameters["n_jobs"] = mp.cpu_count() - 1
        rfreg = get_rf_reg_classifier(hyperparameters=hyperparameters)
        rfreg.fit(X=train_data[features], y=train_data["is_decoy"])

        # Record feature importances
        feature_importances.append(rfreg.feature_importances_)

        # Score predictions
        scores_train = rfreg.score_psms(train[features])
        scores_test = rfreg.score_psms(test[features])
        df.loc[train.index, "prev_score_train"] = scores_train
        df.loc[train.index, "model_score_train"] += scores_train
        df.loc[test.index, "model_score"] = scores_test
    df.loc[:, "model_score_train"] /= len(cv_split_data) - 1

    return df, feature_importances


def train(df, init_eng, sensitivity, q_cut, q_cut_train, n_train, n_eval):
    """Train classifier on input data for a set number of training and evaluation epochs.

    Args:
        df (pd.DataFrame): input data
        init_eng (str): initial engine to rank results by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        q_cut_train (float): q-value cutoff for PSM selection to use during training
        n_train (int): number of training epochs
        n_eval (int): number of evaluation epochs

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits over all eval epochs
        psms (dict): number of top target PSMs found after each epoch

    """
    psms_per_iter = []
    feature_importances = []
    psms = {"train": [], "test": [], "train_avg": None, "test_avg": None}

    # Remove all classifier columns and create safe copy
    df.drop(columns=f"score_processed_rf-reg", errors="ignore", inplace=True)
    df_training = df.copy(deep=True)

    # Create cross-validation splits for training with equal number of spectra
    group_kfold = GroupKFold(n_splits=3)
    groups = df_training["spectrum_id"]
    train_cv_splits = list(group_kfold.split(X=df_training, groups=groups))

    # Record current number of PSMs with q-val < 1%
    psms_per_iter.append(
        calc_num_psms(
            df=df_training,
            score_col=f"score_processed_{init_eng}",
            q_cut=q_cut,
            sensitivity=sensitivity,
        )
    )

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""))
    pbar = tqdm(range(n_train + n_eval))
    for epoch in pbar:
        if epoch == 0:
            # Rank by initial engine's score column during first iteration of training
            score_col = f"score_processed_{init_eng}"
            df_training["model_score_all"] = 0
            df_training["model_score_train_all"] = 0

        else:
            score_col = "prev_score_train"

        df_training, feature_importance_sub = fit_cv(
            df=df_training,
            score_col=score_col,
            cv_split_data=train_cv_splits,
            sensitivity=sensitivity,
            q_cut=q_cut_train,
        )

        # Record how many PSMs are below q-cut in the target set
        psms["train"].append(
            calc_num_psms(
                df=df_training,
                score_col="model_score_train",
                q_cut=q_cut,
                sensitivity=sensitivity,
            )
        )

        psms["test"].append(
            calc_num_psms(
                df=df_training,
                score_col="model_score",
                q_cut=q_cut,
                sensitivity=sensitivity,
            )
        )

        if epoch >= n_train:
            df_training.loc[:, "model_score_train_all"] += df_training[
                "model_score_train"
            ]
            df_training.loc[:, "model_score_all"] += df_training["model_score"]
            feature_importances.extend(feature_importance_sub)

        pbar.set_postfix(
            {"Train PSMs": psms["train"][epoch], "Test PSMs": psms["test"][epoch]}
        )

    logger.remove()
    logger.add(sys.stdout)
    df_training.loc[:, "model_score_train_all"] /= n_eval
    df_training.loc[:, "model_score_all"] /= n_eval
    psms["train_avg"] = calc_num_psms(
        df=df_training,
        score_col="model_score_train_all",
        q_cut=q_cut,
        sensitivity=sensitivity,
    )

    psms["test_avg"] = calc_num_psms(
        df=df_training,
        score_col="model_score_all",
        q_cut=q_cut,
        sensitivity=sensitivity,
    )

    logger.info(
        f"Average PSMs [Train/Test]:\t\t{psms['train_avg']}\t\t{psms['test_avg']}"
    )

    # Show feature importances and deviations for eval epochs
    sigma = np.std(feature_importances, axis=0)
    feature_importances = np.mean(feature_importances, axis=0)
    features = set(df_training.columns).difference(
        set(
            [
                c
                for c in df_training.columns
                for r in knowledge_base.parameters["non_trainable_columns"]
                if c.startswith(r)
            ]
        )
    )
    df_feature_importance = pd.DataFrame(
        {"feature_importance": feature_importances, "standard deviation": sigma},
        index=list(features),
    ).sort_values("feature_importance", ascending=False)
    logger.debug(f"Feature importances:\n{df_feature_importance}")

    # Add averaged scores to df as classifier score
    df.loc[:, "score_processed_peptide_forest"] = df_training["model_score_all"]

    return df, df_feature_importance, psms
