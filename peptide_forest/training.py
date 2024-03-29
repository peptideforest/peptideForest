"""Train peptide forest."""

import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from peptide_forest import knowledge_base
from peptide_forest.regressor_model import RegressorModel


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


def get_feature_cols(df):
    """Get feature columns from dataframe columns.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs

    Returns:
        features (list): list of feature column names
    """
    features = [
        c
        for c in df.columns
        if not any(
            c.startswith(r) for r in knowledge_base.parameters["non_trainable_columns"]
        )
    ]
    return sorted(features)


def fit_cv(df, score_col, cv_split_data, sensitivity, q_cut, conf):
    """Process single-epoch of cross validated training.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): column to score PSMs by
        cv_split_data (list): list with indices of data to split by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        conf (dict): configuration dictionary
        universal_feature_cols (bool):  if training runs on the engine cols directly or
                                        on aggregated columns.

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits in epoch
    """
    # ensure proper None values in config
    for key, value in conf.items():
        if value == "None":
            conf[key] = None

    # Reset scores
    df.loc[:, "model_score"] = 0
    df.loc[:, "model_score_train"] = 0

    feature_importances = []

    for i, split in enumerate(cv_split_data):
        train_inds, test_inds = split
        # Create training data copy for current split
        train = df.loc[train_inds.tolist(), :].copy(deep=True)
        test = df.loc[test_inds.tolist(), :].copy(deep=True)

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
        if train_data[train_data["is_decoy"]].shape[0] < train_targets.shape[0]:
            logger.warning(
                f"More targets below q-value threshold ({train_targets.shape[0]}), "
                f"than decoys available ({train_data[train_data['is_decoy']].shape[0]})"
                f". Sampling from targets."
            )
            train_decoys = train_data[train_data["is_decoy"]]
            train_targets = train_targets.sample(n=len(train_decoys))
        else:
            train_decoys = train_data[train_data["is_decoy"]].sample(
                n=len(train_targets)
            )

        # Combine to form training dataset
        train_data = pd.concat([train_targets, train_decoys]).sample(frac=1)

        # Scale the data
        features = get_feature_cols(df)
        scaler = StandardScaler().fit(train_data.loc[:, features])
        train_data.loc[:, features] = scaler.transform(train_data.loc[:, features])
        train.loc[:, features] = scaler.transform(train.loc[:, features])
        test.loc[:, features] = scaler.transform(test.loc[:, features])

        # Get RF-reg classifier and train
        model = RegressorModel(
            model_type=conf.get("model_type", "random_forest"),
            pretrained_model_path=conf.get("pretrained_model_path", None),
            mode=conf.get("mode", "train"),
            additional_estimators=conf.get("additional_estimators", 50),
            model_output_path=conf.get("model_output_path", None),
            initial_estimators=conf.get("initial_estimators", None),
        )

        model.load()

        model.train(
            X=train_data[features].astype(float),
            y=train_data["is_decoy"].astype(int),
        )

        # Record feature importances
        feature_importances.append(model.get_feature_importances())

        # Score predictions
        scores_train = model.score_psms(train[features].astype(float))
        scores_test = model.score_psms(test[features].astype(float))
        df.loc[train.index, "prev_score_train"] = scores_train
        df.loc[train.index, "model_score_train"] += scores_train
        df.loc[test.index, "model_score"] = scores_test
    df.loc[:, "model_score_train"] /= len(cv_split_data) - 1
    df["prev_score_test"] = df["model_score"]

    model.save()

    return df, feature_importances


def train(df, init_eng, sensitivity, q_cut, q_cut_train, n_train, n_eval, conf):
    """Train classifier on input data for a set number of training and evaluation epochs.

    Args:
        df (pd.DataFrame): input data
        init_eng (str): initial engine to rank results by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        q_cut_train (float): q-value cutoff for PSM selection to use during training
        n_train (int): number of training epochs
        n_eval (int): number of evaluation epochs
        conf (dict): configuration dictionary for training

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
            score_col = "prev_score_test"

        df_training, feature_importance_sub = fit_cv(
            df=df_training,
            score_col=score_col,
            cv_split_data=train_cv_splits,
            sensitivity=sensitivity,
            q_cut=q_cut_train,
            conf=conf,
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
    features = get_feature_cols(df_training)
    df_feature_importance = pd.DataFrame(
        {"feature_importance": feature_importances, "standard deviation": sigma},
        index=list(features),
    ).sort_values("feature_importance", ascending=False)
    logger.debug(f"Feature importances:\n{df_feature_importance}")

    # Add averaged scores to df as classifier score
    df.loc[:, "score_processed_peptide_forest"] = df_training["model_score_all"]

    return df, df_feature_importance, psms
