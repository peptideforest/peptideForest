import peptide_forest
from peptide_forest import runtime, prep
import os
import pandas as pd
import json


def combine_ursgal_csv_files(path_dict):
    """
    Combine separate ursgal search output csv files and return a dataframe. Also output as new csv file (optional).
    Takes a dictionary of input csv files and their respective engine and name of score column.

    Args:
        path_dict (Dict): ursgal path dict
    Returns:
        input_df (pd.DataFrame): combined dataframes
    """

    # List of separate dataframes
    dfs = []

    # Read in all files specified in path_dict, drop columns and append to summarised dataframe
    for file in path_dict.keys():
        slurp_time = runtime.PFTimer()
        slurp_time["slurp"]
        # dtype for column 'Rank' specified to avoid mixed dtypes warning on import
        df = pd.read_csv(file, dtype={"Rank": object})
        df["engine"] = path_dict[file]["engine"]
        df["Score"] = df[path_dict[file]["score_col"]]
        file_output = file.split("/")[-1]
        print(f"Slurping in df for {file_output} in {slurp_time['slurp']}")

        df.drop(
            columns=peptide_forest.knowledge_base.parameteres[
                "columns_to_be_removed_from_input_csvs"
            ],
            errors="ignore",
            inplace=True,
        )
        dfs.append(df)

    input_df = pd.concat(dfs, sort=True).reset_index(drop=True)

    # CF: cast columns based on .... kb?
    input_df["Sequence Post AA"].fillna("-", inplace=True)
    input_df["Sequence Pre AA"].fillna("-", inplace=True)
    input_df["Sequence Start"] = input_df["Sequence Start"].apply(str)
    input_df["Sequence Stop"] = input_df["Sequence Stop"].apply(str)
    input_df["Charge"] = pd.to_numeric(input_df["Charge"], downcast="integer")

    return input_df


def extract_features(
    df,
    cleavage_site,
    min_data,
    read_features_from_cfg=False,
    feature_cols=None,
    path_dict=None,
):
    """
    Calculate features from dataframe containing raw data from a single experiment.

    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        cleavage_site (str): enzyme cleavage site (Currently only "C" implemented and tested)
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
        read_features_from_cfg (bool): Read config/features.json to limit feature_cols
        feature_cols (list of features): List of features to use, note this will be over written by config/feature.json
    Returns:
        df (pd.DataFrame): new dataframe containing the original experiment data and extracted features
        old_cols (List): columns initially in the dataframe
        feature_cols (List): column names of newly calculated features
    """
    # Save columns
    old_cols = df.columns
    if read_features_from_cfg is True:
        features_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "config",
            "features.json",
        )

        if os.path.exists(features_json):
            with open(features_json, "r") as f:
                preset_features = json.load(f)
                if preset_features:
                    feature_cols = preset_features

    elif feature_cols is None:
        feature_cols = []
        for c in old_cols:
            try:

                if pd.to_numeric(df[c]).count() > df.shape[0] * 0.8:
                    feature_cols.append(c)
                    # print("[ok]", c)
                # else:
                #     pass
                # print("[nearly ok]", c)
            except:
                pass
    # print(df.info())
    # for file_path, udict in path_dict.items():
    #     feature_cols.append(udict["score_col"])

    # print(feature_cols)
    # exit(1)
    # Get features and a list of feature names
    df = prep.calc_features(
        df,
        cleavage_site=cleavage_site,
        old_cols=old_cols,
        min_data=min_data,
        feature_cols=feature_cols,
    )

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
    targets = df[df["Is decoy"] == False]
    targets = targets.sort_values(score_col, ascending=False).drop_duplicates(
        "Spectrum ID"
    )

    # Get all the top decoys
    decoys = df[df["Is decoy"] == True]
    decoys = decoys.sort_values(score_col, ascending=False).drop_duplicates(
        "Spectrum ID"
    )

    # Join the data together
    df = pd.concat([targets, decoys])
    df["Is decoy"] = df["Is decoy"].astype(bool)

    return df
