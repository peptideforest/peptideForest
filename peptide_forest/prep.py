import multiprocessing as mp
from functools import partial
from itertools import repeat

import numpy as np
import pandas as pd
import uparma
from loguru import logger
from sklearn.preprocessing import minmax_scale

import peptide_forest.knowledge_base


def _parallel_apply(df, func):
    """
    Maps a function across a dataframe using multiprocessing.
    Args:
        df (pd.DataFrame): input data
        func (method): function to be mapped across the data

    Returns:
        df (pd.DataFrame): input data with function applied
    """
    chunks = np.array_split(df, min(mp.cpu_count() - 1, len(df)))
    with mp.Pool(mp.cpu_count() - 1) as pool:
        df = pd.concat(pool.map(func, chunks))

    return df


def _parallel_calc_delta(df, iterable):
    """
    Calculates data columns using multiprocessing.
    Args:
        df (pd.DataFrame): input data
        iterable (list): columns to be calculated

    Returns:
        df (pd.DataFrame): input data with delta columns appended
    """
    with mp.Pool(mp.cpu_count() - 1) as pool:
        single_delta_cols = list(pool.starmap(calc_delta, zip(repeat(df), iterable)))
        if len(single_delta_cols) == 0:
            logger.warning(
                "No score columns fulfil conditions for delta column calculation. "
                "Proceeding without delta columns."
            )
            return df
        new_cols = pd.concat(single_delta_cols, axis=1)

    return pd.concat([df, new_cols], axis=1)


def add_stats(stats, df):
    """
    Adds score stats to dataframe.
    Args:
        stats (dict): stats (min/max) for relevant columns
        df (pd.DataFrame): input data

    Returns:
        df (pd.DataFrame): input data with delta columns appended
    """
    df["_score_min"] = df.apply(
        lambda row: stats[row["Search Engine"]]["min_score"], axis=1
    )
    df["_score_max"] = df.apply(
        lambda row: stats[row["Search Engine"]]["max_score"], axis=1
    )

    return df


def check_mass_sanity(df):
    """
    Checks for consistent mass values across unique PSMs.
    Args:
        df (pd.DataFrame): input data

    Returns:
        (bool): False, if masses are unique
    """
    return (
        (
            df.groupby(["Spectrum ID", "Sequence", "Charge", "Modifications"]).agg(
                {"uCalc m/z": "nunique", "Exp m/z": "nunique"}
            )
            != 1
        )
        .any(axis=0)
        .any()
    )


def calc_delta(df, delta_col):
    """
    Computes score difference of highest and 2nd/3rd highest score in a spectrum for a given engine.
    Args:
        df (pd.DataFrame): df
        delta_col (str): name of the delta column to be calculated (e.g delta_score_3_engine_B)

    Returns:
        df (pd.DataFrame): input data with delta columns appended

    """
    # Get information from new column name
    n = int(delta_col.split("_")[2])
    eng = delta_col[14:]
    score_col = f"Score_processed_{eng}"

    # Find nth highest score per spectrum
    nth_score = (
        df.sort_values(score_col, ascending=False)
        .groupby("Spectrum ID")[score_col]
        .nth(n=(n - 1))
    )
    # Set to nan if 0
    nth_score.replace(0.0, pd.NA, inplace=True)
    # Calculate different to all other scores in group
    deltas = df.sort_values(score_col)[score_col] - df.sort_values(score_col)[
        "Spectrum ID"
    ].map(nth_score)

    # Return expanded column to multiprocessing pool
    return deltas.rename(delta_col)


def get_stats(df):
    """
    Collects engine-level stats and applies modifications for score calculation.
    Args:
        df (pd.DataFrame): input data

    Returns:
        stats (dict): keys are engines with values being dicts describing min and max possible scores.
    """
    # Collect stats with min and max scores on engine-level and process
    min_max_stats = df.groupby("Search Engine").agg({"Score": ["min", "max"]})["Score"]
    stats = {
        eng: {
            "min_score": 1e-30 if "omssa" in eng and min_score < 1e-30 else min_score,
            "max_score": max_score,
        }
        for eng, min_score, max_score in zip(
            min_max_stats.index, min_max_stats["min"], min_max_stats["max"]
        )
    }

    return stats


def calc_col_features(df, min_data=0.7):
    """
    Compute all column level features for input data.
    Args:
        df (pd.DataFrame): input data
        min_data (float): fraction of PSMs with higher number of i PSMs per spectrum id
                          and engine than total number of spectra per engine

    Returns:
        df (pd.DataFrame): input data with added column level features
    """
    # Determine delta columns to calculate
    delta_columns = []
    size = df.groupby(["Search Engine", "Spectrum ID"]).size().droplevel(1)
    d2 = (size >= 2).groupby("Search Engine").agg("sum") / (size > 0).groupby(
        "Search Engine"
    ).agg("sum")
    d3 = (size >= 3).groupby("Search Engine").agg("sum") / (size > 0).groupby(
        "Search Engine"
    ).agg("sum")
    delta_columns += [f"delta_score_2_{col}" for col in d2[d2 >= min_data].index]
    delta_columns += [f"delta_score_3_{col}" for col in d3[d3 >= min_data].index]

    # Convert all scores so that a higher score is better
    udict = uparma.UParma()
    engines = df["Search Engine"].unique()
    bigger_score_translations = udict.get_default_params("unify_csv_style_1")[
        "bigger_scores_better"
    ]["translated_value"]
    bigger_score_better_engs = [
        bigger_score_translations.get(e, False) for e in engines
    ]
    scores_that_need_to_be_inverted = [
        c for c, bsb in zip(engines, bigger_score_better_engs) if bsb is False
    ]
    inds = df[df["Search Engine"].isin(scores_that_need_to_be_inverted)].index
    df.loc[inds, "Score_processed"] = -np.log10(df.loc[inds, "Score_processed"])
    df.loc[:, "Score_processed"] = df.groupby("Search Engine")[
        "Score_processed"
    ].transform(lambda x: minmax_scale(x))

    # Collect columns used in indices
    core_idx_cols = [
        # "Spectrum Title",
        "Spectrum ID",
        "Sequence",
        "Modifications",
        "Is decoy",
        "Protein ID",
        "Charge",
    ]
    value_cols = ["Score_processed"]
    remaining_idx_cols = [
        c
        for c in df.columns
        if not c in core_idx_cols + value_cols and c != "Search Engine"
    ]

    # Pivot
    df = pd.pivot_table(
        df,
        index=core_idx_cols + remaining_idx_cols,
        values=value_cols,
        columns="Search Engine",
    )
    df.columns = ["_".join(t) for t in df.columns.to_list()]
    df.reset_index(inplace=True)

    # Note reported PSMs and fill scores
    score_cols = [c for c in df.columns if "Score_processed_" in c]
    for col in score_cols:
        df.loc[:, f"reported_by_{col.replace('Score_processed_', '')}"] = ~df[
            col
        ].isna()
    df.fillna({col: 0.0 for col in score_cols}, inplace=True)

    # Calculate delta columns
    df = _parallel_calc_delta(df, delta_columns)
    # Fill missing values with minimum for each column
    df.fillna({col: df[col].min() for col in delta_columns}, inplace=True)

    return df


def calc_row_features(df):
    """
    Compute all row level features for input data.
    Args:
        df (pd.DataFrame): input data

    Returns:
        df (pd.DataFrame): input data with added row level features
    """
    # Get stats
    stats = get_stats(df=df)

    # Add stats columns and process scores accordingly
    mp_add_stats = partial(add_stats, stats)
    df = _parallel_apply(df, mp_add_stats)
    df["Score_processed"] = df["Score"]

    # Check for consistent masses
    if check_mass_sanity(df):
        raise ValueError(
            "Recorded mass values are deviating across engines for identical matches. Rerunning ursgal is recommended."
        )

    # More data based training features
    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].str.split(r"<\|>|;").apply(set).apply(len)

    # Drop columns that are no longer relevant
    df.drop(
        columns=peptide_forest.knowledge_base.parameters["remove_after_row_features"],
        inplace=True,
    )

    return df
