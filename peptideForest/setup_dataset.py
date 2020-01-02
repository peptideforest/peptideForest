import time

from peptideForest import runtime, prep

import pandas as pd
import numpy as np


def combine_ursgal_csv_files(
    path_dict, output_file,
):
    """
    Combine separate ursgal search output csv files and return a dataframe. Also output as new csv file (optional).
    Takes a dictionary of input csv files and their respective engine and name of score column.

    Args:
        path_dict (str): path to ursgal path dict as .json
        output_file (str): path to save new data frame to, do not save if None (default)
    Returns:
        input_df (pd.DataFrame): combined dataframes
    """

    # List of separate dataframes
    dfs = []

    # Name columns to be disregarded
    cols_to_drop = [
        "Spectrum Title",
        "Mass Difference",
        "Raw data location",
        "Rank",  # [TRISTAN] note: only XTandem, which doesn have rank .... will rename it in the next release?
        "Calc m/z",  # too buggy
    ]

    # Read in all files specified in path_dict, drop columns and append to summarised dataframe
    for file in path_dict.keys():
        slurp_time = runtime.PFTimer()
        slurp_time["slurp"]
        # [TRISTAN] added to avoid warning? weil manchmal 2 values and ";"??
        df = pd.read_csv(file, dtype={"Rank": object})
        df["engine"] = path_dict[file]["engine"]
        df["Score"] = df[path_dict[file]["score_col"]]
        file_output = file.split("/")[-1]
        print(f"Slurping in df for {file_output} in", "{slurp}".format(**slurp_time))

        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
        dfs.append(df)

    input_df = pd.concat(dfs, sort=True).reset_index(drop=True)

    if output_file is not None:
        input_df.to_csv(output_file)

    return input_df


# [TRISTAN] defaults allow for reduction: input; combine_eng=True; keep_ursgal = false
def extract_features(df):
    """
    Calculate features from dataframe containing raw data from a single experiment.

    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Returns:
        df (pd.DataFrame): new dataframe containing the original experiment data and extracted features
        old_cols (List): columns initially in the dataframe
        feature_cols (List): column names of newly calculated features
    """
    # [TRISTAN]
    # output_df = input_df
    # combine make_dataset and get_features
    # prep.calc_features

    # Save columns
    old_cols = df.columns

    # Get features and a list of feature names
    df = prep.calc_features(df, old_cols)

    feature_cols = list(set(df.columns) - set(old_cols))
    q_value_cols = [f for f in df.columns if "q-value" in f]
    feature_cols = [f for f in feature_cols if f not in q_value_cols]

    # Replace missing scores with 0
    score_cols = [f for f in df.columns if "Score_processed" in f]
    df[score_cols] = df[score_cols].fillna(0)

    return df, old_cols, feature_cols


def get_top_target_decoy(df, score_col):
    """
    Remove all PSMs except the top target and the top decoy for each spectrum from the dataset.
    Args:
        df (pd.DataFrame): dataframe containing feature values
        score_col (str): column name to rank PSMs by

    Returns:
        df (pd.DataFrame): input dataframe with non-top targets/decoys removed

    """
    # Get all the top targets
    # [TRISTAN] why why why? cant i use df[~df["Is decoy].astype(bool)] -> weil dtype -> rausfinden warum es vorher ging
    targets = df[df["Is decoy"] == False]
    targets = targets.sort_values(score_col, ascending=False).drop_duplicates(
        "Spectrum ID"
    )

    # Get all the top decoys
    decoys = df[df["Is decoy"]]
    decoys = decoys.sort_values(score_col, ascending=False).drop_duplicates(
        "Spectrum ID"
    )

    # Determine sample size
    n_sample = len(decoys)

    decoys = decoys.sample(n=n_sample, replace=False)

    # Join the data together
    df = pd.concat([targets, decoys])
    df["Is decoy"] = df["Is decoy"].astype(bool)

    return df
