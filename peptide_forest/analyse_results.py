"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Analyse results after training
"""

# pylint: disable=unused-import
from typing import Any, Dict, Pattern, Set, Tuple, List

import numpy as np
import pandas as pd

from .models import ml_methods
from .fit_models import get_q_vals, find_psms_to_keep


def get_top_targets(df: pd.DataFrame, q_val_cut: float = 0.01) -> pd.DataFrame:
    """
  Identify which PSMs in a dataframe are top-targets (i.e. is the highest ranked target
  PSM for a given spectrum with q-value<i%). Returns results for all Score_processed columns.
  Arguments:
    - df: dataframe with q-values for each PSM
    - q_val_cut: q-value to use as a cut off, as a fraction
  Returns:
    - same dataframe with new columns indicating if the PSM is a top-target
  """

    # first drop any top-target columns that are already in the dataframe
    cols = [c for c in df.columns if "top_target" in c]
    df = df.drop(cols, axis=1)

    # get the column names
    cols_q_val = df.columns[df.columns.str[0:8] == "q-value_"]
    cols_target = ["top_target" + c.split("q-value")[-1] for c in cols_q_val]

    for col_target, col_q_val in zip(cols_target, cols_q_val):
        df_toptargets = df[(df[col_q_val] <= q_val_cut) & (~df["Is decoy"])]
        df_toptargets = df_toptargets.sort_values(col_q_val)
        df_toptargets = df_toptargets.drop_duplicates("Spectrum ID")
        df[col_target] = False
        df.loc[df_toptargets.index, col_target] = True

    return df


def get_ranks(
    df: pd.DataFrame, method: str = "first", from_scores: bool = True
) -> pd.DataFrame:
    """
  Add a column with the rank of each PSM for all different Score_processed columns.
  Arguments:
    - df: dataframe with scores for each PSM
    - methods: which method to use (based on pandas rank method). Default is first,
               such that PSMs with the same score are ranked on the order they appear
               in the dataframe.
    - from_scores: rank by scores (True, default), or from q-values (False)
  Returns:
    - same dataframe with new columns indicating the ranks
  """

    if from_scores:
        # get the column names
        cols_score = df.columns[df.columns.str[0:16] == "Score_processed_"]
        cols_rank = ["rank" + c.split("Score_processed")[-1] for c in cols_score]
        # get the rankd for each score_processed column
        for col_score, col_rank in zip(cols_score, cols_rank):
            df[col_rank] = df[col_score].rank(ascending=False, method=method)
    else:
        # get the column names
        cols_q_val = df.columns[df.columns.str[0:8] == "q-value_"]
        cols_rank = ["rank" + c.split("q-value")[-1] for c in cols_q_val]
        # get the rankd for each score_processed column
        for col_q_val, col_rank in zip(cols_q_val, cols_rank):
            df[col_rank] = df[col_q_val].rank(ascending=True, method=method)
    return df


def get_shifted_psms(
    df: pd.DataFrame, x_name: str, y_name: str, n_return: int = 10
) -> pd.DataFrame:
    """
  Make dataframes showing which PSMs were top-targets before training but no longer are,
  and vice-versa. 
  Arguments:
    - df:  dataframe with training data and analysed results
    - x_name: name of method used for baseline (e.g. search engine name)
    - y_name: name of method used for comparison (e.g. ML model name)
    - n_return: number of examples to return. Returns the n_return most extreme examples,
                i.e. those with the biggest jump in ranking. Default is 10
  Returns:
    - df_new_top_targets: dataframe containing information on the new top-targets
    - df_old_top_targets: dataframe containing information on the old top-targets
  """
    col_x = f"rank_{x_name}"
    tt_x = f"top_target_{x_name}"

    tt_y = f"top_target_{y_name}"

    # non top targets that are now top targets
    df_new_top_targets = (
        df[~df[tt_x] & df[tt_y]].sort_values(col_x, ascending=False).copy(deep=True)
    )
    df_new_top_targets = df_new_top_targets.reset_index()
    print(
        f"Number non top targets for {x_name} that are now top targets: {len(df_new_top_targets)}"
    )
    if n_return is not None:
        df_new_top_targets = df_new_top_targets.head(n_return)

    # up_rank_for_spectrum: previously was not top PSM for that spectrum
    df_new_top_targets["up_rank_for_spectrum"] = False
    for i in df_new_top_targets.index:
        spectrum_id = df_new_top_targets.loc[i, "Spectrum ID"]
        sequence = df_new_top_targets.loc[i, "Sequence"]
        protein_id = str(df_new_top_targets.loc[i, "Protein ID"])
        mods = df_new_top_targets.loc[i, "Modifications"]
        df_spectrum = df[df.loc[:, "Spectrum ID"] == spectrum_id]
        if (df_spectrum[f"Score_processed_{x_name}"] != 0).all():
            df_spectrum = df_spectrum.sort_values(
                f"Score_processed_{x_name}", ascending=False
            )
            if any(
                [
                    sequence != df_spectrum["Sequence"].values[0],
                    protein_id != df_spectrum["Protein ID"].astype(str).values[0],
                    mods != df_spectrum["Modifications"].values[0],
                ]
            ):
                df_new_top_targets.loc[i, "up_rank_for_spectrum"] = True

    # top targets that are now not top targets
    df_old_top_targets = (
        df[df[tt_x] & ~df[tt_y]].sort_values(col_x, ascending=True).copy(deep=True)
    )
    df_old_top_targets = df_old_top_targets.reset_index()
    print(
        f"Number top targets for {x_name} that are now not top targets: {len(df_old_top_targets)}"
    )
    if n_return is not None:
        df_old_top_targets = df_old_top_targets.head(n_return)

    # down_rank_for_spectrum: moved down the rankings for this spectrum
    # new_best_psm_is_top_target: spectrum has new best match that is also a top target

    df_old_top_targets["down_rank_for_spectrum"] = False
    df_old_top_targets["new_best_psm_is_top_target"] = False

    for i in df_old_top_targets.index:
        spectrum_id = df_old_top_targets.loc[i, "Spectrum ID"]
        sequence = df_old_top_targets.loc[i, "Sequence"]
        protein_id = str(df_old_top_targets.loc[i, "Protein ID"])
        mods = df_old_top_targets.loc[i, "Modifications"]
        df_spectrum = df[df.loc[:, "Spectrum ID"] == spectrum_id]
        if (df_spectrum[f"Score_processed_{x_name}"] != 0).all():
            df_spectrum = df_spectrum.sort_values(
                f"Score_processed_{y_name}", ascending=False
            )
            if any(
                [
                    sequence != df_spectrum["Sequence"].values[0],
                    protein_id != df_spectrum["Protein ID"].astype(str).values[0],
                    mods != df_spectrum["Modifications"].values[0],
                ]
            ):
                df_old_top_targets.loc[i, "down_rank_for_spectrum"] = True
                df_old_top_targets.loc[i, "new_best_psm_is_top_target"] = df_spectrum[
                    f"top_target_{y_name}"
                ].values[0]

    return df_new_top_targets, df_old_top_targets


def calc_final_q_vals(
    df: pd.DataFrame,
    col: str,
    fac: float = 0.9,
    top_psm_only: bool = True,
    from_method: str = None,
) -> pd.DataFrame:
    """
  Calculate the q-values for a given score column
  Arguments:
    - df: dataframe with training results
    - col: name of method to calulcate q-values for
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    - top_psm_only: keep only highest scoring PSM for each spectrum, default=True
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - same dataframe with q-values added as a column
  """
    if "Score_processed_" in col:
        col = col.split("Score_processed_")[-1]
    score_col = f"Score_processed_{col}"
    q_col = f"q-value_{col}"
    df_scores = get_q_vals(
        df, score_col, fac=fac, top_psm_only=top_psm_only, from_method=from_method
    )
    df[q_col] = 1.0
    df.loc[df_scores.index, q_col] = df_scores["q-value"]
    max_q_val = df.loc[df_scores.index, q_col].max()
    df[q_col] = df[q_col].fillna(max_q_val)
    return df


def calc_all_final_q_vals(
    df: pd.DataFrame,
    fac: float = 0.9,
    top_psm_only: bool = True,
    from_method: str = None,
) -> pd.DataFrame:
    """
  Calculate the q-values for all score columns
  Arguments:
    - df: dataframe with training results
    - fac: estimate of fraction of True Positives in target dataset (default = 0.9)
    - top_psm_only: keep only highest scoring PSM for each spectrum, default=True
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - same dataframe with q-values added as new columns
  """
    # get a list of all the score columns
    cols = [c for c in df.columns if "Score_processed_" in c]
    for col in cols:
        df = calc_final_q_vals(
            df, col, fac=fac, top_psm_only=top_psm_only, from_method=from_method
        )
    return df


def get_protein_maxarea_dataset(df: pd.DataFrame, prot: str) -> pd.DataFrame:
    """
  Get the max area values for a given protein, for each peptide associated with that
  protein. Values for each method (search engine + ML methods) are returned.
  Arguments:
    - df: dataframe with final results are max_area values
    - prot: name of protein ID
  Returns:
    - df_maxarea: maxarea values for each peptide
  """
    df_prot = df[df["protein"] == prot]
    df_prot["avg_maxarea"] = (
        df_prot["MAXAREA_msgfplus"].groupby(df_prot["Sequence"]).transform("mean")
    )
    df_prot = df_prot.sort_values("avg_maxarea", ascending=False)
    cols = ["protein", "ISOTOPELABEL_ID", "Sequence", "avg_maxarea"]

    cols_max_area = [c for c in df_prot.columns if "MAXAREA_" in c]
    cols_max_area = [c.split("cols_max_area")[-1] for c in cols_max_area]

    df_maxarea = None

    for c in cols_max_area:

        if df_maxarea is not None:
            df_maxarea_sub = df_prot[cols + [f"MAXAREA_{c}"]]
            df_maxarea_sub.columns = cols + ["MAXAREA"]
            df_maxarea_sub["method"] = c
            df_maxarea = pd.concat([df_maxarea, df_maxarea_sub])
        else:
            df_maxarea = df_prot[cols + [f"MAXAREA_{c}"]]
            df_maxarea.columns = cols + ["MAXAREA"]
            df_maxarea["method"] = c

    return df_maxarea


def get_num_psms_by_method(
    df: pd.DataFrame,
    q_val_cut: float = 0.01,
    methods: List = None,
    from_method: str = None,
) -> pd.DataFrame:
    """
  Return a dataframe giving number of PSMs identified as top targets for each method
  that results are present in input dataframe
  Arguments:
    - df: dataframe containing results from search engines and ML training
    - methods: list of methods to use, if None, use all methods
    - q_val_cut: q-value used to identify top-targets
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - df_num_psms: dataframe containing number of PSMs with q_val < q_val_cut
                   for each method
  """

    # get a list of methods
    # q-values from percoaltor are ignored, as these are not comparible to our results
    if methods is None:
        methods = [c for c in df.columns if "top_target" in c]
        methods = [c for c in methods if "ursgal" not in c and "from" not in c]

    # calculate q-values
    df = calc_all_final_q_vals(df, from_method=from_method)
    # flag which are top targets
    df = get_top_targets(df, q_val_cut=q_val_cut)
    df_num_psms = df[
        [c for c in df.columns if "top_target_" in c and c in methods]
    ].sum()
    df_num_psms = df_num_psms.to_frame().reset_index()
    df_num_psms.columns = ["method", "n_psms"]
    df_num_psms["method"] = df_num_psms["method"].apply(
        lambda x: x.split("top_target_")[-1]
    )
    engine_cols = [m for m in df.columns if "top_target_" in m]
    engine_cols = [
        m for m in engine_cols if all(ml_method not in m for ml_method in ml_methods)
    ]

    if "any" not in df_num_psms["method"].values:
        df_num_psms = df_num_psms.append(
            {"method": "any-engine", "n_psms": df[engine_cols].any(axis=1).sum()},
            ignore_index=True,
        )
    if "all" not in df_num_psms["method"].values:
        df_num_psms = df_num_psms.append(
            {"method": "all-engines", "n_psms": df[engine_cols].all(axis=1).sum()},
            ignore_index=True,
        )
    if "majority" not in df_num_psms["method"].values:
        n_majority_engines = np.ceil(0.5 * len(engine_cols))
        n_psms = (df[engine_cols].sum(axis=1) >= n_majority_engines).sum()
        df_num_psms = df_num_psms.append(
            {"method": "majority-engines", "n_psms": n_psms}, ignore_index=True
        )
    df_num_psms = df_num_psms.sort_values("n_psms", ascending=False)
    return df_num_psms


def get_num_psms_against_q_cut(
    df: pd.DataFrame,
    methods: List = None,
    q_val_cuts: List = None,
    from_method: str = None,
) -> pd.DataFrame:
    """
  Return a dataframe containing number of top-target PSMs against q-value
  used as the cut-off to identify top-targets.
  Arguments:
    - df: dataframe containing results of search engines and training
    - methods: list of methods to use, if None, use all methods
    - q_val_cuts: list of q-values to use, default is None (use values between
                  1e-4 and 1e-1)
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
  Returns:
    - df_num_psms_q: dataframe containing number of PSMs at each q-value cut-off
                     for each method in df
  """

    # make sure q-values are calculated
    df = calc_all_final_q_vals(df, from_method=from_method)

    # get the q-value list
    if q_val_cuts is None:
        q_val_cuts = sorted(
            [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(4, 1, -1)]
        ) + [1e-1]

    # get a list of methods
    # q-values from percoaltor are ignored, as these are not comparible to our results
    if methods is None:
        methods = [c for c in df.columns if "top_target" in c]
        methods = [c for c in methods if "ursgal" not in c and "from" not in c]
    # initiate the dataframe for the results
    df_num_psms_q = pd.DataFrame(
        index=[str(q) for q in q_val_cuts],
        columns=methods + ["top_target_any-engine", "top_target_all-engines"],
    )
    # get a list of the engine columns
    engine_cols = [
        f"top_target_{m}" for m in ["msgfplus", "mascot", "xtandem", "omssa"]
    ]
    for q_val_cut in q_val_cuts:
        # get the top-targets for this q_value_cut-off
        df = get_top_targets(df, q_val_cut=q_val_cut)
        # calculate the number of q-values for this cut-off
        df_num_psms = df[[c for c in df.columns if "top_target_" in c]].sum()
        df_num_psms_q.loc[str(q_val_cut), :] = df_num_psms[df_num_psms_q.columns]

        # calculate results for any engine identifies a PSM as a top-target
        tmp = df[engine_cols].any(axis=1)
        df_num_psms_q.loc[str(q_val_cut), "top_target_any-engine"] = sum(tmp)
        # calculate results for all-engines in agreement
        tmp = df[engine_cols].all(axis=1)
        df_num_psms_q.loc[str(q_val_cut), "top_target_all-engines"] = sum(tmp)
        # calculate results for majorty-engines in agreement
        n_majority_engines = np.ceil(0.5 * len(engine_cols))
        tmp = df[engine_cols].sum(axis=1) >= n_majority_engines
        df_num_psms_q.loc[str(q_val_cut), "top_target_majority-engines"] = sum(tmp)
    return df_num_psms_q


def get_removed_top_targets(
    df: pd.DataFrame, method: str, output_file: str = None
) -> pd.DataFrame:
    """
  Return a dataframe containing high scoring PSMs (with score > the minimum score for a
  top target) for a given method.
  Arguments:
    - df: dataframe containing training results
    - method: method to use (i.e. search engine or machine learning method)
    - output_file: filename to save output dataframe to. If None (default), don't save
  Returns:
    - df_removed_top_targets: dataframe containing target PSMs that are removed from
                              rankings (if another target or decoy has the same core),
                              but with scores that would make them top targets
  """
    # column names for scores and top_targets
    col_score = f"Score_processed_{method}"
    col_target = f"top_target_{method}"
    # get the minimum score for top_targets
    min_score = df.loc[df[col_target], col_score].min()
    # get PSMs that are removed due to having the same score as other PSMs
    # (for a given spectrum)
    df = find_psms_to_keep(df, col_score)
    # get a dataframe of all targets that would be removed but have scores that would
    # make them top targets
    df_removed_top_targets = df[
        (~df["keep in"]) & (~df["Is decoy"]) & (df[col_score] >= min_score)
    ]
    # save the dataframe if filename is provided
    if output_file:
        if df_removed_top_targets.empty:
            print(f"Dataframe is empty, not savaing to:\n{output_file}")
        else:
            df_removed_top_targets.to_csv(output_file)
    return df_removed_top_targets
