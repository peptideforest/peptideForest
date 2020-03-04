import re
import warnings
import numpy as np
import pandas as pd

# Regex
ENGINES = {
    "mascot": ("Mascot:Score", False),
    "msgfplus": ("MS-GF:SpecEValue", True),
    "omssa": ("OMSSA:evalue", True),
    "xtandem": (r"X\!Tandem:hyperscore", False),
    "msfragger": (r"MSFragger:Hyperscore", False),
}
DELIM_REGEX = re.compile(r"<\|>|;")
PROTON = 1.00727646677
CLEAVAGE_SITES = {"R", "K", "-"}


def test_cleavage_aa(
    aa_field, aa_start,
):
    """
    Test whether pre/post amino acid is consistent with enzyme cleavage site.
    Args:
        aa_field (str): (multiple) pre/post amino acids
        aa_start (str): start of sequence
    Returns:
        (bool): True if amino acid is consistent with cleavage site, False otherwise
    """
    all_aas = set(re.split(DELIM_REGEX, aa_field))
    return any(aa in CLEAVAGE_SITES for aa in all_aas) or aa_start in ["1", "2"]


def test_sequence_aa_c(
    aa, post_aa,
):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
    Args:
        aa (str): start/end of sequence
        post_aa (str): (multiple) pre/post amino acids
    Returns:
        (bool): True if start/end is consistent with cleavage site, False otherwise
    """
    all_post_aas = set(re.split(DELIM_REGEX, post_aa))
    return aa in CLEAVAGE_SITES or "-" in all_post_aas


# only if cleavage site included
# def test_sequence_aa_n(
#     aa, aa_start,
# ):
#     """
#     Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
#     Args:
#         aa (str): start/end of sequence
#         aa_start (str): start of sequence
#     Returns:
#         (bool): True if start/end is consistent with cleavage site, False otherwise
#     """
#     return aa in CLEAVAGE_SITES or aa_start in [1, 2]


def parse_protein_ids(protein_id,):
    """
    Turns ursgal dataframe column "Protein ID" into a list of all protein IDs.
    Args:
        protein_id (str): separated ProteinIDs
    Returns:
        prot_id_set (: True if start/end is consistent with cleavage site, False otherwise
    """
    prot_id_set = set(re.split(DELIM_REGEX, protein_id))
    return prot_id_set


def transform_score(score, score_stats):
    """
    Transforms a score to a base 10 logarithmic range.
    Args:
        score (float): score value from engine
        engine (str): database search engine that generated score
        minimum_score (float): used when score is negative or 0
    Returns:
        score (float): transformed score
    """

    if score_stats["max_score"] < 1:
        if score <= score_stats["min_score"]:
            transformed_score = -np.log10(score_stats["min_score"])
        else:
            # score can get very small, set to -log10(1e-30) if less than 1e-30
            transformed_score = -np.log10(score)

    else:
        transformed_score = score
    return transformed_score


def calc_delta_score_i(
    df, i, min_data,
):
    """
    Calculate delta_score_i, which is the difference in score between a PSM and the ith ranked PSM for a given
    spectrum. It is calculated for targets and decoys combined. It is only calculated when the fraction of
    spectra with more than i PSMs  is greater than min_data. Missing values are replaced by the mean.
    It is calculated for each engine.
    Args:
        df (pd.DataFrame): ursgal dataframe
        i (int): rank to compare to (i.e. i=2 -> subtract score of 2nd ranked PSM)
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
    Returns:
        df (pd.DataFrame): ursgal dataframe with delta_score_i added
    """
    # Name of the new column
    col = f"delta_score_{i}"
    decoy_state = col + "_to_decoy"
    delta_type = col + "_delta_type"

    # Initialize to nan (for PSMs from different engines)
    df[col] = np.nan
    df[decoy_state] = np.nan
    df[delta_type] = np.nan

    for engine in df["engine"].unique():

        # Get data for engine
        df_engine = df[df["engine"] == engine]

        # Get number of PSMs for each spectrum ID
        psm_counts = df_engine["Spectrum ID"].value_counts()

        # Test if there enough spectra with more than i target and i decoy PSMs
        if len(psm_counts[psm_counts >= i]) / len(psm_counts) > min_data:
            inds = df_engine.loc[
                df_engine["Spectrum ID"].isin(psm_counts[psm_counts >= i].index), :
            ].index
            ith_best = df_engine.loc[inds, :].groupby("Spectrum ID")
            ith_best_indices = ith_best["Score_processed"].transform(
                lambda x: x.nlargest(i).idxmin()
            )

            ith_best_value = df.loc[ith_best_indices]["Score_processed"]
            ith_best_is_decoy = df.loc[ith_best_indices]["Is decoy"]
            ith_best_value.index = inds
            ith_best_is_decoy.index = inds

            df.loc[inds, col] = df.loc[inds, "Score_processed"] - ith_best_value
            df.loc[inds, decoy_state] = ith_best_is_decoy

            mean_val = df.loc[inds, col].mean()
            # Replace missing with mean
            inds = df_engine.loc[
                df_engine["Spectrum ID"].isin(psm_counts[psm_counts < i].index), :
            ].index
            df.loc[inds, col] = mean_val

    # New columns indicating delta_type where:
    # 0 = target -> decoy or decoy -> target
    # 1 = target -> target or decoy -> decoy

    df[decoy_state] = df[decoy_state].astype(bool)
    if not all(df["delta_score_2"].isna()):

        df.loc[
            (~df["Is decoy"] & df[decoy_state]) | (df["Is decoy"] & ~df[decoy_state]),
            delta_type,
        ] = 0
        df.loc[
            (~df["Is decoy"] & ~df[decoy_state]) | (df["Is decoy"] & df[decoy_state]),
            delta_type,
        ] = 1

    df = df.drop(columns=decoy_state)

    return df


def preprocess_df(df):
    """
    Preprocess ursgal dataframe:
    Remove decoy PSMs overlapping with targets and fill missing modifications (None).
    Sequences containing 'X' are removed. Missing mScores (= 0) are replaced with minimum value for mScore.
    Operations are performed inplace on dataframe!
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Returns:
        df (pd.DataFrame): preprocessed dataframe
    """
    # Fill missing modifications
    df["Modifications"].fillna("None", inplace=True)

    decoys = df[df["Is decoy"]]
    targets = df[~df["Is decoy"]]

    seqs_in_both = set(targets["Sequence"]) & set(decoys["Sequence"])
    if len(seqs_in_both) > 0:
        warnings.warn("Target and decoy sequences contain overlaps.", Warning)
    df.drop(df[df["Sequence"].isin(seqs_in_both)].index, axis=0, inplace=True)

    # Remove Sequences with "X"
    df.drop(
        labels=df[df.Sequence.str.contains("X") == True].index, axis=0, inplace=True
    )

    # Missing mScores are replaced by minimum value for mScore in data
    if "mScore" in df.columns:
        min_mscore = df[df["mScore"] != 0]["mScore"].min()
        df.loc[df["mScore"] == 0, "mScore"] = min_mscore

    return df


def get_stats(df):
    """
    Calculate minimum scores across all PSMs for all engines.
    Ignores OMSSA scores lower than 1e-30.
    Args:
        df (pd.DataFrame): ursgal dataframe
    
    Returns:
        (Dict of str: Any): Dict of engines containing dict of min scores.
    """
    # fixed code - thanks to test [cf]

    stats = {}
    engines = df["engine"].unique()
    # df['Score_-log10'] = -np.log10(df['Score'])
    for engine in engines:
        stats[engine] = {}
        engine_df = df[df["engine"] == engine]

        # print(engine_df)
        # Minimum score (transformed)
        # CF: wrong does not set miniumum value to 1e-30 for OMSSA
        # stats[engine]["min_score"] = -np.log10(
        #     engine_df.loc[engine_df["Score"] >= 1e-30, "Score"].min()
        # )
        # CF": corrected:
        # 1e-30 and omssa only
        if engine == "omssa" and engine_df.Score.min() < 1e-30:
            stats[engine]["min_score"] = 1e-30
        else:
            stats[engine]["min_score"] = engine_df.Score.min()
        stats[engine]["max_score"] = engine_df.Score.max()
    return stats


def combine_engine_data(
    df, feature_cols,
):
    """
    Calculate row-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): dataframe containing search engine results
        feature_cols (list): list of calculated feature columns in dataframe
    Returns:
        df (pd.DataFrame):  dataframe with results for each search engine combined for each individual
                            experimental-theoretical PSM.
    """

    # Get a list of columns that will be different for each engine.
    # Mass based columns can be slightly different between engines. The average is taken at the end.
    cols_single = list(
        [
            "Score_processed",
            "delta_score_2",
            "delta_score_3",
            "delta_score_2_delta_type",
            "delta_score_3_delta_type",
            "Mass",
        ]
    )

    # Get a list of columns that should be the same for each engine
    cols_same = list(sorted([f for f in feature_cols if f not in cols_single]))

    # Columns to group by
    cols_u = ["Spectrum ID", "Sequence", "Modifications", "Protein ID", "Is decoy"]

    cols = cols_u + cols_same + cols_single

    # Initialize the new dataframe
    df_combine = pd.DataFrame(columns=cols)

    # Go through each engine and get the results
    for engine in df["engine"].unique():
        df_engine = df.loc[df["engine"] == engine, cols]
        # Rename the columns that will have different names
        cols_single_engine = [f"{c}_{engine}" for c in cols_single]
        df_engine.columns = cols_u + cols_same + cols_single_engine

        # Merge results for each engine together (or start the dataframe)
        if df_combine.empty:
            df_combine = df_engine.copy(deep=True)
        else:
            df_combine = df_combine.merge(
                df_engine, how="outer", on=(cols_u + cols_same)
            )

    # Drop columns that are all NaNs
    df_combine = df_combine.dropna(axis=1, how="all")

    # Drop columns that contain all the same result
    breakpoint()
    df_combine = df_combine.drop(
        [c for c in df_combine.columns if len(df_combine[c].unique()) == 1], axis=1
    )

    # Drop rows that are identical
    df_combine = df_combine.drop_duplicates()

    # Average mass based columns and drop the engine specific ones
    eng_names = [engine for engine in df["engine"].unique()]
    cols = [f"Mass_{eng_name}" for eng_name in eng_names]
    df_combine["Mass"] = df_combine[cols].mean(axis=1)
    df_combine = df_combine.drop(cols, axis=1)

    return df_combine


def row_features(df, cleavage_site="C", proton=1.00727646677, max_charge=None):
    """
    Calculate row-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        cleavage_site (str): enzyme cleavage site (Currently only "C" implemented and tested)
        proton (float): used for mass calculation and is kwargs for testing purpose
    Returns:
        df (pd.DataFrame): input dataframe with added row-level features for each PSM
    """
    stats = get_stats(df)

    # Calculate processed score
    df["Score_processed"] = df.apply(
        lambda row: transform_score(row["Score"], stats[row["engine"]]), axis=1,
    )

    df["Mass"] = (df["uCalc m/z"] - proton) * df["Charge"]

    # Only works with trypsin for now
    if cleavage_site == "C":
        df["enzN"] = df.apply(
            lambda x: test_cleavage_aa(x["Sequence Pre AA"], x["Sequence Start"]),
            axis=1,
        )
        df["enzC"] = df.apply(
            lambda x: test_sequence_aa_c(x["Sequence"][-1], x["Sequence Post AA"]),
            axis=1,
        )

    else:
        raise ValueError("Only cleavage sites consistent with trypsin are accepted.")

    df["enzInt"] = df["Sequence"].str.count(r"[R|K]")
    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].apply(parse_protein_ids).apply(len)

    # Get maximum charge to use for columns
    if max_charge is None:
        max_charge = df["Charge"].max()

    # Create categorical charge columns
    for i in range(1, max_charge):
        # pd.to_numeric(df['Sequence Start'], downcast="integer")
        df[f"Charge{i}"] = (df["Charge"] == i).astype(int)
        df[f"Charge{i}"] = df[f"Charge{i}"].astype("category")
    df[f">Charge{max_charge}"] = (df["Charge"] >= max_charge).astype(int)
    return df


def col_features(df, min_data=0.7):
    """
    Calculate col-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
    Returns:
        df (pd.DataFrame): input dataframe with added col-level features for each PSM
    """
    # delta_score_2: difference between first and second score for a spectrum.
    # If < min_data have two scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=2, min_data=min_data)

    # delta_score_3: difference between first and third score for a spectrum.
    # If < min_data have three scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=3, min_data=min_data)

    # log of the number of times the peptide sequence for a spectrum is found in the set
    df["lnNumPep"] = df.groupby("Sequence")["Sequence"].transform("count").apply(np.log)

    return df


def calc_features(df, cleavage_site, old_cols, min_data, feature_cols):
    """
    Main function to calculate features from unified ursgal dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        cleavage_site (str): enzyme cleavage site (Currently only "C" implemented and tested)
        old_cols (List): columns initially in the dataframe
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
        feature_cols (List): column names of newly calculated features


    Returns:
        df (pd.DataFrame): input dataframe with added features for each PSM
    """
    df = preprocess_df(df)
    df = row_features(df, cleavage_site=cleavage_site)
    df = col_features(df, min_data=min_data)
    if not feature_cols:
        feature_cols = list(set(df.columns) - set(old_cols))
    else:
        engines = df["engine"].unique().tolist()
        for e in engines:
            feature_cols = [feature.split("_" + e)[0] for feature in feature_cols]
        feature_cols = list(dict.fromkeys(feature_cols))

    df = combine_engine_data(df, feature_cols)

    return df
