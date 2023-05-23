"""Train peptide forest."""
import sys
import types

import numpy as np
import pandas as pd
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


def get_classifier(alg, hyperparameters):
    """Initialize random forest regressor.

    Args:
        alg (str): algorithm to use for training  one of ["random_forest_scikit",
                    "random_forest_xgb", "xgboost", "adaboost"]
        hyperparameters (dict): hyperparameters for classifier

    Returns:
        clf (xgboost.XGBRFRegressor): classifier with added method to score PSMs
    """
    if alg == "random_forest_scikit":
        clf = RandomForestRegressor(**hyperparameters)
    elif alg == "random_forest_xgb":
        clf = XGBRFRegressor(**hyperparameters)
    elif alg == "xgboost":
        clf = XGBRegressor(**hyperparameters)
    elif alg == "adaboost":
        clf = AdaBoostRegressor(**hyperparameters)
    else:
        logger.error(
            f"Algorithm {alg} not supported. Choose one of [randomforest_xgb',"
            f" 'random_forest_scikit, 'xgboost', 'adaboost']. Defaulting to "
            f"'random_forest_scikit'."
        )
        clf = RandomForestRegressor(**hyperparameters)

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


def get_highest_scoring_engine(df):
    """Find engine with the highest number of target PSMs under 1% q-val

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs

    Returns:
        init_eng (str): name of engine with highest number of target PSMs under 1% q-val
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


def fit_cv(df, score_col, sensitivity, q_cut, model, scaler, epoch, algorithm):
    """Process single-epoch of cross validated training.

    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): column to score PSMs by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        model (xgboost.XGBRegressor): model to iteratively train

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits in epoch
        model (xgboost.XGBRegressor): trained model
        cycle_results (dict): dictionary with performance indicators for each cycle
    """
    feature_importances = []

    features = get_feature_columns(df)

    # todo: check why this seems to be such a bad idea :(
    if epoch > 100:
        # score PSMs with pretrained model
        df["model_score"] = model.score_psms(df[features].astype(float))
        score_col = "model_score"

    # Use only top target and top decoy per spectrum
    train_data = (
        df.sort_values(score_col, ascending=False)
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
    # test_decoys = train_data[train_data["is_decoy"]].drop(train_decoys.index)
    # test_decoys.to_csv("test_decoys.csv", mode='a', header=False, index=False)

    # Combine to form training dataset
    train_data = pd.concat([train_targets, train_decoys]).sample(frac=1)

    # Scale the data
    features = get_feature_columns(train_data)
    if epoch == 0:
        scaler = StandardScaler().fit(train_data.loc[:, features])
    train_data.loc[:, features] = scaler.transform(train_data.loc[:, features])
    df.loc[:, features] = scaler.transform(df.loc[:, features])

    # create train test split
    # todo: remove astype... hack
    X_train, X_test, y_train, y_test = train_test_split(
        train_data[features].astype(float),
        train_data["is_decoy"].astype(float),
        test_size=0.2,
        random_state=42,
    )

    # Train the model
    if epoch == 0:
        # Define the hyperparameter grid
        param_grid = {
            # "learning_rate": [0.01],
            "max_depth": [3, 7, 15, 30],
            "n_estimators": [10, 50, 100, 200],
        }

        # Initialize the GridSearch object
        grid_search = GridSearchCV(model, param_grid, cv=3, verbose=2, n_jobs=-1)

        # Fit the GridSearch to the data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # Train the model using the best parameters
        model = get_classifier(alg=algorithm, hyperparameters=best_params)

        # Train initial model
        model.fit(X=X_train, y=y_train)
    elif epoch > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data[features].astype(float),
            train_data["is_decoy"].astype(float),
            test_size=0.99,
            random_state=42,
        )
    else:
        model.fit(X=X_train, y=y_train, xgb_model=model)

    # Record feature importances
    feature_importances.append(model.feature_importances_)

    # Score predictions
    # todo: remove astype... hack
    scores_train = model.score_psms(df[features].astype(float))
    df.loc[:, "prev_score_train"] = scores_train

    # Score test predictions
    y_pred = model.score_psms(X_test)

    y_test = 2 * (0.5 - y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
    cycle_results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    return df, feature_importances, model, scaler, cycle_results


def train(
    gen,
    sensitivity,
    q_cut,
    q_cut_train,
    n_train,
    algorithm,
    max_mp_count=None,
):
    """Train classifier on input data for a set number of training and evaluation epochs.

    Args:
        gen (generator yielding pd.DataFrames): input data
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection
        q_cut_train (float): q-value cutoff for PSM selection to use during training
        n_train (int): number of training epochs
        algorithm (str): algorithm to use for training  one of ["random_forest_scikit",
                            "random_forest_xgb", "xgboost", "adaboost"]
        max_mp_count (int): maximum number of processes to use for training

    Returns:
        df (pd.DataFrame): dataframe with training columns added
        feature_importances (list): list of arrays with the feature importance for all splits over all eval epochs
        psms (dict): number of top target PSMs found after each epoch
        model (xgboost.XGBRegressor): trained model to be used for inference
        classifier_test_performance (dict): dictionary with performance indicators for each epoch

    """
    feature_importances = []
    psms = {"train": [], "test": [], "train_avg": None, "test_avg": None}
    classifier_test_performance = {}
    scaler = None

    # Get classifier
    hyperparameters = knowledge_base.parameters[f"{algorithm}_hyperparameters"]
    if algorithm != "adaboost":
        hyperparameters["n_jobs"] = max_mp_count
    model = get_classifier(alg=algorithm, hyperparameters=hyperparameters)

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""))
    pbar = tqdm(range(n_train))

    # todo: remove hack
    df = next(gen)
    df.drop(columns=f"score_processed_rf-reg", errors="ignore", inplace=True)

    for epoch in pbar:
        # try:
        #     df = next(gen)
        # except StopIteration:
        #     break
        # df.drop(columns=f"score_processed_rf-reg", errors="ignore", inplace=True)
        df_training = df.sample(frac=1, random_state=42).copy(deep=True)
        # df_training = df.copy(deep=True)
        score_col = get_highest_scoring_engine(df_training)

        df_training, feature_importance_sub, new_model, scaler, kpis = fit_cv(
            df=df_training,
            score_col=score_col,
            sensitivity=sensitivity,
            q_cut=q_cut_train,
            model=model,
            scaler=scaler,
            epoch=epoch,
            algorithm=algorithm,
        )
        model = new_model
        classifier_test_performance[epoch] = kpis

        # Record how many PSMs are below q-cut in the target set
        psms["train"].append(
            calc_num_psms(
                df=df_training,
                score_col="prev_score_train",
                q_cut=q_cut,
                sensitivity=sensitivity,
            )
        )

        # Record feature importances
        feature_importances.extend(feature_importance_sub)

        pbar.set_postfix({"Train PSMs": psms["train"][epoch]})

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

    return df, df_feature_importance, psms, model, scaler, classifier_test_performance
