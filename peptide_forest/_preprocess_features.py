"""
Copyright © 2018 by minds.ai, Inc.
All rights reserved

Data pre-processing from a Ursgal unified csv
"""
from typing import Any, Dict, Pattern, Set, Tuple, List

import re
import numpy as np
import pandas as pd

ENGINES = {
    "mascot": ("Mascot:Score", False),
    "msgfplus": ("MS-GF:SpecEValue", True),
    "omssa": ("OMSSA:evalue", True),
    "xtandem": (r"X\!Tandem:hyperscore", False)
    #             ^-- score field        ^-- will score be transformed ?
}
AA_DELIM_REGEX = re.compile(r"<\|>|\;")
PROTON = 1.00727646677  # type: float
CLEAVAGE_SITES = set(["R", "K", "-"])


def test_cleavage_aa(
    aa_field, aa_start=None, delim_regex=AA_DELIM_REGEX, cleavage_sites=CLEAVAGE_SITES
):
    """
    Test whether pre/post amino acid is consistent with enzyme cleavage site
    Args:
      aa_field: a string containing (multiple) pre/post amino acids
      aa_start: start of sequence
      delim_regex: the regex to separate multiple amino acids
      cleavage_sites: the amino acids at which the enzyme cleaves

    Returns: True if one amino acid is consistent with cleavage site, False otherwise
    """
    all_aas = set(re.split(delim_regex, aa_field))
    return any(aa in cleavage_sites for aa in all_aas) or aa_start in ["1", "2"]


def test_sequence_aa_c(aa, pre_post_aa, cleavage_sites=None):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if is cut at end
    Args:
      aa: a string containing start/end of sequence
      pre_post_aa: a string containing (multiple) pre/post amino acids
      cleavage_sites: the amino acids at which the enzyme cleaves

    Returns: True if start/end is consistent with cleavage site, False otherwise
    """
    xx = 1
    cleavage_sites = cleavage_sites or CLEAVAGE_SITES
    return aa in cleavage_sites or "-" in pre_post_aa


def test_sequence_aa_n(aa, aa_start, cleavage_sites=None):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if is cut at end
    Args:
      aa: a string containing start/end of sequence
      aa_start: start of sequence
      cleavage_sites: the amino acids at which the enzyme cleaves

    Returns: True if start/end is consistent with cleavage site, False otherwise
    """
    xx = 1

    cleavage_sites = cleavage_sites or CLEAVAGE_SITES
    return aa in cleavage_sites or aa_start in ["1", "2"]


def parse_protein_ids(protein_id, sep="<|>"):
    """
    Turns the unified CSV column "Protein ID"
    into a list of all protein IDs.
    Args:
      protein_id: protein IDs separated by 'sep'
      sep: the separator between proteins
    """
    xx = 1
    clean = protein_id.replace("decoy_", "").strip()
    prot_id_set = list(clean.split(sep))
    return prot_id_set


def transform_score(score, engine, minimum_score):
    """
    Transforms a score to a base 10 logarithmic range
    Args:
      score: A score value from an engine
      engine: The database search engine that produced the score
      minimum_score: Used when score is negative or 0
    """
    if "_" in engine:
        eng = engine.split("_")[0]
    else:
        eng = engine

    if eng not in ENGINES:
        raise ValueError(f"Engine {engine} not known")
        # NOTE: this has to be part of the file_dict

    elif ENGINES[eng][1]:

        if score > 0:
            transformed_score = -np.log10(score)
        else:
            transformed_score = minimum_score
        return transformed_score
    return score


def preprocess_df(df, exchange_L_to_I=True):
    """
    Preprocess unified Ursgal csv (as dataframe)
    - maps amino acid isomers to single value (I)
    - removes decoy psms that overlap with targets
    - divides Exp m/z and Calc m/z by charge for Omssa results

    Note: Operations are in place on dataframe

    Args:
    df: the unified Ursgal csv (as dataframe)

    Return:
        df (pd.DataFrame): original inplace modified dataframe
    """
    if exchange_L_to_I:
        df["Sequence"] = df["Sequence"].str.replace("L", "I")

    targets = df[df["Is decoy"]]
    decoys = df[~df["Is decoy"]]

    # overlap in Sequences between targets/decoys seems to only happen for mascot
    df.drop(
        decoys[decoys["Sequence"].isin(targets["Sequence"].unique())].index,
        inplace=True,
    )

    # fix Exp m/z and Calc m/z for omssa
    inds = df[df["engine"] == "omssa_2_1_9"].index
    df.loc[inds, "Exp m/z"] = df.loc[inds, "Exp m/z"] / df.loc[inds, "Charge"]
    df.loc[inds, "Calc m/z"] = df.loc[inds, "Calc m/z"] / df.loc[inds, "Charge"]
    return df


def get_stats(df):
    """
    Calculate statistics of unified Ursgal csv (as dataframe)
    Args:
        df: the unified Ursgal csv (as dataframe)
    Returns:
        stats (dict): in the form of::
            {
                engine: { 'min_score': minimum score across all psms for engine}
            }
    """
    stats = {}

    engines = df["engine"].unique()
    for engine in engines:
        stats[engine] = {}

        # Minimum score (transformed)
        stats[engine]["min_score"] = -np.log10(
            df[(df["engine"] == engine) & (df["Score"] > 0)]["Score"].min()
        )

    return stats


def row_features(df, stats, max_charge=None, cleavage_site="C"):
    """
    Calculate row-level features from unified csv (as dataframe)
    Adds features as columns in input dataframe in place
    Args:
        df: the unified Ursgal csv (as dataframe)
        stats: dictionary containing minimum score for each engine
        max_charge: maximum charge to use for charge catagorical columns.
            If charge > max_charge, add to '>max_charge+1' column.
            Default = None (use maximum from dataset)
    Returns:
        df (pd.DataFrame): dataframe with features added
    """
    df["Score_processed"] = df.apply(
        lambda row: transform_score(
            row["Score"], row["engine"], stats[row["engine"]]["min_score"]
        ),
        axis=1,
    )

    df["Mass"] = (df["Exp m/z"] * df["Charge"]) - (df["Charge"] - 1) * PROTON
    df["Delta_m/z"] = df["Calc m/z"] - df["Exp m/z"]
    # Note: this should use uCalc as more accurate ...

    df["absDelta_m/z"] = df["Delta_m/z"].apply(np.absolute)

    # get the log of delta mass and replace values that give log(0) with minimum
    # Note: ...
    log_min = np.log(df.loc[df["absDelta_m/z"] > 0, "absDelta_m/z"].min())
    df["lnabsDelta_m/z"] = df["absDelta_m/z"].apply(
        lambda x: np.log(x) if x != 0 else log_min
    )

    if cleavage_site == "C":
        df["enzN"] = df.apply(
            lambda row: test_cleavage_aa(row["Sequence Pre AA"], row["Sequence Start"]),
            axis=1,
        )
        df["enzC"] = df.apply(
            lambda x: test_sequence_aa_c(x["Sequence"][-1], x["Sequence Post AA"]),
            axis=1,
        )
    elif cleavage_site == "N":
        df["enzC"] = df.apply(lambda x: test_cleavage_aa(x["Sequence Post AA"]), axis=1)
        # ^-- not sure if that is right ...
        df["enzN"] = df.apply(
            lambda x: test_sequence_aa_n(x["Sequence"][0], x["Sequence Start"]), axis=1
        )

    df["enzInt"] = df.apply(
        lambda row: sum(1 for aa in row["Sequence"] if aa in CLEAVAGE_SITES), axis=1
    )

    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].apply(parse_protein_ids).apply(len)

    # get maximum charge to use for columns
    max_charge = max_charge or df["Charge"].max(axis=0)
    # make catagorical charge columns
    for i in range(max_charge):
        df[f"Charge{i+1}"] = df["Charge"] == i + 1
    df[f">Charge{max_charge+1}"] = df["Charge"] > max_charge
    return df


def get_top_targets_decoys(df, balance_dataset=False):
    """
    Get the top target and top decoy for each Spectrum ID based on score. This is done for each engine
    Args:
    df: the unified Ursgal csv (as dataframe)
    balance_dataset: which targets and decoys to keep.
                     - False (default): keep top target and decoy for each PSM
                     - True: keep top target and decoy for each PSM, only for PSMs where there is
                             one of each
    Returns:
    df: the unified Ursgal csv (as dataframe), with only top target and decoys
    """

    dfs_engine = {
        engine: df[df["engine"] == engine] for engine in df["engine"].unique()
    }
    df = None

    for df_engine in dfs_engine.values():
        # get top targets
        targets = df_engine[~df_engine["Is decoy"]]
        targets = targets.sort_values(
            "Score_processed",
            ascending=False
            # NOTE: Does that always work if greater score is better ?
        ).drop_duplicates(["Spectrum ID"])

        # get top decoys
        decoys = df_engine[df_engine["Is decoy"]]
        decoys = decoys.sort_values("Score_processed", ascending=False).drop_duplicates(
            ["Spectrum ID"]
        )

        # merge them together
        df_engine = pd.concat([targets, decoys]).sort_index()

        if balance_dataset is True:
            # only keep those where we have one target and one decoy
            spec_id_counts = df_engine[df_engine["Spectrum ID"].value_counts() == 2]
            spec_id_counts = df_engine["Spectrum ID"].value_counts()
            spec_id_counts = spec_id_counts[spec_id_counts == 2]
            df_engine = df_engine.loc[
                df_engine["Spectrum ID"].isin(spec_id_counts.index), :
            ]

        if df:
            df = pd.concat([df, df_engine.copy(deep=True)]).sort_index()
        else:
            df = df_engine.copy(deep=True)

    return df


# calculate targets+decoys combined
def calc_delta_score_i(df, i, min_data):
    """
    Calculate delta_score_i, which is the difference in score between a each PSM and the
    ith ranked PSM for a given spectra. Is calculated for Targets and Decoys combined.
    It is only calculated when the fraction of spectra with more than i PSMs is greater
    than min_data. Missing values are replaced by the mean. This is calculated for each engine.
    Arguments:
    - df: the unified Ursgal csv (as dataframe)
    - i: rank to compare to (if = 2, then subract score of 2nd ranked PSM, etc.)
    - min_data: minimum fraction of spectra for which we require that there are at least i PSMs
    Returns:
    - df: the unified Ursgal dataframe with delta_score_i added
    """
    col = f"delta_score_{i}"
    # name of the new column

    for engine in df["engine"].unique():

        # get data for engine
        df_engine = df[df["engine"] == engine]

        # get number of PSMs for each spectrum ID
        psm_counts = df_engine["Spectrum ID"].value_counts()

        # test if there enough spectra with more than i target and i decoy PSMs
        if len(psm_counts[psm_counts >= i]) / len(psm_counts) > min_data:

            # initialize to nan (for PSMs from different engines)
            df[col] = np.nan

            inds = df_engine.loc[
                df_engine["Spectrum ID"].isin(psm_counts[psm_counts >= i].index), :
            ].index
            ith_best = df_engine.loc[inds, :].groupby("Spectrum ID")
            ith_best = ith_best["Score_processed"].transform(
                lambda x: x.nlargest(i).min()
            )
            df.loc[inds, col] = df.loc[inds, "Score_processed"] - ith_best
            mean_val = df.loc[inds, col].mean()
            # replace missing with mean
            inds = df_engine.loc[
                df_engine["Spectrum ID"].isin(psm_counts[psm_counts < i].index), :
            ].index
            df.loc[inds, col] = mean_val

    return df


def col_features(df, only_top=False, balance_dataset=False):
    """
    Calculate column-level features from unified Ursgal csv (as dataframe)
    Adds features as columns in input dataframe in place,
    and updates stats with score_list per decoy/target and spectrum
    Args:
    df: the unified Ursgal csv (as dataframe)
    only_top: only return the top target and the top decoy for each spectrum ID, default = False
    balance_dataset: which targets and decoys to keep.
                     - False (default): keep top target and decoy for each PSM
                     - True: keep top target and decoy for each PSM, only for PSMs where there is
                             one of each
    Returns:
    df: dataframe with additional column features
    """

    # delta_score_2: difference between first and second score for a spectrum.
    # If < 90% have two scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=2, min_data=0.9)

    # delta_score_3: difference between first and third score for a spectrum.
    # If < 75% have three scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=3, min_data=0.75)

    # get the top target and top decoy for each spectrum
    if only_top:
        df = get_top_targets_decoys(df, balance_dataset=balance_dataset)

    # log of the number of times the peptide sequence for a spectrum is found in the set
    df["lnNumPep"] = df.groupby("Sequence")["Sequence"].transform("count").apply(np.log)

    # calculation is slow, needs speeding up
    # log of the number of times the most common protein for a spectrum is found in the set
    # proteins = pd.DataFrame([p for d in df['Protein ID list'] for p in d], columns=['Protein'])
    # protein_counts = proteins['Protein'].value_counts()
    # df['lnNumProt'] =
    # np.log(df['Protein ID list'].apply(lambda prots: protein_counts.loc[prots].max()))
    return df


def combine_engine_data(
    df: pd.DataFrame, feature_cols: List, keep_ursgal: bool = False
) -> pd.DataFrame:
    """
  Combine the results for each search engine. Group by individual experimental-theoretical PSMs.
  Arguments:
    - df: dataframe containing search engine results
    - feature_cols: list of calculated features
    - keep_ursgal: keep original ursgal q-values (if present in dataframe)
  Returns:
    - df_combine: dataframe with results for each search engnine combined for each individual
                  experimental-theoretical PSM.
  """

    # get a list of columns that will be different for each engine.
    # mass based columns can be slightly different between engines. The average is taken at the end.
    cols_single = list(
        [
            "Score_processed",
            "delta_score_2",
            "delta_score_3",
            "Mass",
            "Delta_m/z",
            "absDelta_m/z",
            "lnabsDelta_m/z",
        ]
    )
    if keep_ursgal:
        # also keep the q-value columns
        df = df.rename({"q-value": "q-value_ursgal"}, axis=1)
        cols_single += ["q-value_ursgal"]

    # get a list of columns that should be the same for each engine
    cols_same = list(sorted([f for f in feature_cols if f not in cols_single]))

    # columns to group by
    cols_u = [
        "Spectrum ID",
        "Sequence",
        "Sequence Start",
        "Sequence Stop",
        "Modifications",
        "Is decoy",
        "Protein ID",
    ]

    cols = cols_u + cols_same + cols_single

    # initialize the new dataframe
    df_combine = pd.DataFrame(columns=cols)

    # go through each engine and get the results
    for engine in df["engine"].unique():
        df_engine = df.loc[df["engine"] == engine, cols]
        eng_names = engine.split("_")[0]
        # rename the columns that will have different names
        cols_single_engine = [f"{c}_{eng_names}" for c in cols_single]
        df_engine.columns = cols_u + cols_same + cols_single_engine

        # merge results for each engine together (or start the dataframe)
        if df_combine.empty:
            df_combine = df_engine.copy(deep=True)
        else:
            df_combine = df_combine.merge(
                df_engine, how="outer", on=(cols_u + cols_same)
            )

    # drop columns that are all NaNs
    df_combine = df_combine.dropna(axis=1, how="all")

    # drop columns that contain all the same result
    df_combine = df_combine.drop(
        [c for c in df_combine.columns if len(df_combine[c].unique()) == 1], axis=1
    )

    # drop rows that are identical
    df_combine = df_combine.drop_duplicates()

    # average mass based columns and drop the engine specific ones
    for col in ["Mass", "Delta_m/z", "absDelta_m/z", "lnabsDelta_m/z"]:
        eng_names = [engine.split("_")[0] for engine in df["engine"].unique()]
        cols = [f"{col}_{eng_name}" for eng_name in eng_names]
        df_combine[col] = df_combine[cols].mean(axis=1)
        df_combine = df_combine.drop(cols, axis=1)

    if keep_ursgal:
        q_value_cols = [f for f in df_combine.columns if "q-value" in f]
        q_value_cols_new = [f.split("ursgal_")[-1] for f in q_value_cols]
        q_value_cols_new = [f"q-value_ursgal-{f}" for f in q_value_cols_new]
        df_combine = df_combine.rename(
            {q_old: q_new for q_old, q_new in zip(q_value_cols, q_value_cols_new)},
            axis=1,
        )

    return df_combine


def calc_features(
    df,
    only_top=False,
    balance_dataset=False,
    combine_engines=False,
    pkl_path=None,
    compression="bz2",
    max_charge=None,
    keep_ursgal=False,
):
    """
    Main function to calculate features from unified Ursgal csv format (as dataframe)
    and optionally store dataframe as pickle

    Args:
    - df: dataframe of unified Ursgal csv format
    - only_top: only return the top target and the top decoy for each spectrum ID, default = False
    - balance_dataset: which targets and decoys to keep.
                       - False (default): keep top target and decoy for each PSM
                       - True: keep top target and decoy for each PSM, only for PSMs where there is
                               one of each
    - combine_engines: combine PSMs for all engines
    - pkl_path: output dataframe to pickle file path
    - compression: pickle compression, for normal pickle set to None
    - max_charge: maximum charge to use for charge catagorical columns. If charge > max_charge,
                  add to '>max_charge+1' column. Default = None (use maximum from dataset)
    - keep_ursgal: keep original ursgal q-values (if present in dataframe)
    Returns:
    - df: dataframe with features for each PSM
    """
    org_cols = set(df.columns)
    df = df.astype({"Exp m/z": float, "Calc m/z": float})
    df = preprocess_df(df)
    stats = get_stats(df)
    df = row_features(df, stats, max_charge=max_charge)
    df = col_features(df, only_top=only_top, balance_dataset=balance_dataset)

    if combine_engines:
        feature_cols = list(set(df.columns) - org_cols)
    df = combine_engine_data(df, feature_cols, keep_ursgal=keep_ursgal)

    if pkl_path is not None:
        df.to_pickle(pkl_path, compression=compression)

    return df
