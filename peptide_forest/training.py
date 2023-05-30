"""Train peptide forest."""
import sys
import types

import numpy as np
import pandas as pd
import xgboost
from loguru import logger
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRFRegressor, XGBRegressor

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


def _adjust_q_cut(df, targets, q_vals, q_cut, min_targets=10, catch_decoys=True):
    """Adjust q-value cutoff to ensure minimum number of targets."""

    while len(targets) < min_targets:
        q_cut += 0.01
        logger.info(f"q_cut too low, increasing to {q_cut}")
        q_cut_met_targets = q_vals.loc[
            (q_vals["q-value"] <= q_cut) & (~q_vals["is_decoy"])
        ].index
        targets = df.loc[q_cut_met_targets, :]

    if len(targets) > len(df[df["is_decoy"]]):
        if catch_decoys:
            logger.warning(
                f"Number of targets ({len(targets)}) exceeds number of decoys ("
                f"{len(df[df['is_decoy']])}). Sampling targets to match number of "
                f"decoys."
            )
            return targets.sample(n=len(df[df["is_decoy"]]))
        else:
            raise ValueError(
                f"Number of targets ({len(df)}) exceeds number of decoys "
                f"({len(df[df['is_decoy']])})."
            )

    return targets


def _filter_targets(df, score_col, sensitivity, q_cut, dynamic_q_cut=True):
    train_q_vals = calc_q_vals(
        df=df,
        score_col=score_col,
        sensitivity=sensitivity,
        top_psm_only=False,
        get_fdr=False,
        init_score_col=None,
    )[["q-value", "is_decoy"]]
    train_q_cut_met_targets = train_q_vals.loc[
        (train_q_vals["q-value"] <= q_cut) & (~train_q_vals["is_decoy"])
    ].index
    train_targets = df.loc[train_q_cut_met_targets, :]

    if dynamic_q_cut:
        return _adjust_q_cut(
            df=df, targets=train_targets, q_vals=train_q_vals, q_cut=q_cut
        )
    else:
        return train_targets


def get_classifier(hyperparameters):
    """Initialize random forest regressor.

    Args:
        alg (str): algorithm to use for training  one of ["random_forest_scikit",
                    "random_forest_xgb", "xgboost", "adaboost"]
        hyperparameters (dict): hyperparameters for classifier

    Returns:
        clf (xgboost.XGBRegressor): classifier with added method to score PSMs
    """
    clf = xgboost.XGBRegressor(**hyperparameters)

    # Add scoring function
    clf.score_psms = types.MethodType(_score_psms, clf)

    return clf


def get_feature_columns(df):
    """Get names of columns to be used as features.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs

    Returns:
        features (list): list of column names to be used as features
    """
    features = set(df.columns).difference(
        set(
            [
                c
                for c in df.columns
                for r in knowledge_base.parameters["non_trainable_columns"]
                if c.startswith(r)
            ]
        )
    )
    return list(features)


def _generate_train_data(df, score_col, sensitivity, q_cut):
    """Generate training data for classifier by getting best targets and matching decoys"""
    # Use only top target and top decoy per spectrum
    train_data = (
        df.sort_values(score_col, ascending=False)
        .drop_duplicates(["spectrum_id", "is_decoy"])
        .copy(deep=True)
    )

    # Filter for target PSMs with q-value < q_cut
    train_targets = _filter_targets(
        df=train_data,
        score_col=score_col,
        sensitivity=sensitivity,
        q_cut=q_cut,
        dynamic_q_cut=True,
    )

    # Get same number of decoys to match targets at random
    train_decoys = train_data[train_data["is_decoy"]].sample(n=len(train_targets))

    # Combine to form training dataset
    return pd.concat([train_targets, train_decoys]).sample(frac=1)


def get_highest_scoring_engine(df):
    """Find engine with the highest number of target PSMs under 1% q-val

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs

    Returns:
        init_eng (str): name of engine with the highest number of target PSMs under 1% q-val
    """
    psms_per_eng = {}
    score_processed_cols = [c for c in df if "score_processed_" in c]
    for eng_score in score_processed_cols:
        psms_per_eng[eng_score.replace("score_processed_", "")] = calc_num_psms(
            df,
            score_col=eng_score,
            q_cut=0.01,
            sensitivity=0.9,
        )
    logger.debug(f"PSMs per engine with q-val < 1%: {psms_per_eng}")

    init_eng = max(psms_per_eng, key=psms_per_eng.get)
    logger.info(
        f"Training from {init_eng} with {psms_per_eng[init_eng]} top target PSMs"
    )
    return f"score_processed_{init_eng}"


def fit_model(X, y, model=None, hyperparameters=None):
    if hyperparameters is None:
        try:
            hyperparameters = model.get_params()
        except AttributeError:
            hyperparameters = knowledge_base.parameters["hyperparameters"]
            logger.warning(
                "No hyperparameters provided, using values in knowledge" "base."
            )
        logger.info("No hyperparameters provided, using values from prev model.")
    if model is not None and model.get_params() == hyperparameters:
        logger.warning(
            "Hyperparameters provided match those of model. Model will not"
            "be updated in training"
        )
    clf = get_classifier(hyperparameters=hyperparameters)
    clf.fit(X=X, y=y, xgb_model=model)
    return clf


def update_hyperparameters(model):
    params = model.get_params()
    params["n_estimators"] += 10
    params["max_depth"] += 1
    params["learning_rate"] *= 0.5
    return params


def train(
    gen,
    sensitivity,
    q_cut,
    q_cut_train,
    n_train,
    max_mp_count=None,
):
    """Train classifier on input data for a set number of training and evaluation epochs.

    Args:
        gen (generator yielding pd.DataFrames): input data
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        q_cut_train (float): q-value cutoff for PSM selection to use during training
        n_train (int): number of training epochs
        max_mp_count (int): maximum number of processes to use for training

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits over all eval epochs
        psms (dict): number of top target PSMs found after each epoch
        model (xgboost.XGBRegressor): trained model to be used for inference
    """
    feature_importances = []
    psms = {"train": [], "test": [], "train_avg": None, "test_avg": None}

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""))
    pbar = tqdm(range(n_train))

    for epoch in pbar:
        try:
            df = next(gen)
            df_training = df.copy(deep=True)
        except StopIteration:
            break

        df.drop(columns=f"score_processed_rf-reg", errors="ignore", inplace=True)

        # scoring
        score_col = get_highest_scoring_engine(df_training)

        # get train data
        features = get_feature_columns(df)
        train_data = _generate_train_data(df, score_col, sensitivity, q_cut_train)

        # Scale and transform data
        scaling_mode = "first"  # todo: could also be "global", "batch", None
        if scaling_mode == "first" and epoch == 0:
            scaler = StandardScaler().fit(train_data.loc[:, features])
        elif scaling_mode == "batch":
            scaler = StandardScaler().fit(train_data.loc[:, features])

        train_data.loc[:, features] = scaler.transform(train_data.loc[:, features])
        df.loc[:, features] = scaler.transform(df.loc[:, features])

        X = train_data.loc[:, features].astype(float)
        y = train_data["is_decoy"].astype(float)

        param_grid = {
            "max_depth": [3, 6, 9],
            "n_estimators": [20, 50, 100],
            "learning_rate": [0.01, 0.05],
            "min_child_weight": [1, 3],
            "reg_lambda": [10000],
        }

        # training
        if epoch == 0:
            # initial fit
            # find the best hyperparameters & return eval of best model
            # grid search for best params

            hyperparameters = knowledge_base.parameters["hyperparameters"]  # todo
            grid_search = GridSearchCV(
                estimator=get_classifier(hyperparameters=hyperparameters),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2,
                scoring="neg_mean_squared_error",
            )

            # Fit the grid search to the data
            grid_search.fit(X, y)
            hyperparameters = grid_search.best_params_

            hyperparameters["n_jobs"] = max_mp_count
            logger.info(f"Epoch {epoch}, using hyperparameters: {hyperparameters}")
            model = fit_model(
                X=X,
                y=y,
                model=None,
                hyperparameters=hyperparameters,
            )
        else:
            # iterative fit
            # find the best hyperparameters & return eval of best model
            param_grid = {
                "n_estimators": list(
                    range(model.n_estimators, model.n_estimators + 50, 5)
                ),
                "reg_lambda": [model.reg_lambda / 2],
            }
            grid_search = GridSearchCV(
                estimator=get_classifier(hyperparameters=hyperparameters),
                param_grid=param_grid,
                cv=3,
                # todo: for n_jobs -1 i do get errors, n_jobs=1 is awfully slow. Fix...
                n_jobs=1,
                verbose=2,
                scoring="neg_mean_squared_error",
            )
            grid_search.fit(X, y, xgb_model=model)
            hyperparameters = grid_search.best_params_
            logger.info(f"Epoch {epoch}, using hyperparameters: {hyperparameters}")

            # hyperparameters = knowledge_base.parameters["hyperparameters"]  # todo
            # hyperparameters["n_jobs"] = max_mp_count

            # hyperparameters = update_hyperparameters(model)
            model = fit_model(
                X=X,
                y=y,
                model=model,
                hyperparameters=hyperparameters,
            )

        # Record feature importances
        feature_importances.extend(model.feature_importances_)

        pbar.set_postfix()  # todo: add info

    logger.remove()
    logger.add(sys.stdout)

    # Show feature importances and deviations for eval epochs
    sigma = np.std(feature_importances, axis=0)
    feature_importances = np.mean(feature_importances, axis=0)
    features = get_feature_columns(df=df_training)
    df_feature_importance = pd.DataFrame(
        {"feature_importance": feature_importances, "standard deviation": sigma},
        index=list(features),
    ).sort_values("feature_importance", ascending=False)
    logger.debug(f"Feature importances:\n{df_feature_importance}")

    return df, df_feature_importance, psms, model, scaler
