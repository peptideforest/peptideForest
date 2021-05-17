import re
import warnings

import click
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
    aa_field,
    aa_start,
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
    aa,
    post_aa,
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


def mass_sanity_check(df):
    new_df = df.copy(deep=True)
    already_warned = False
    for name, grp in new_df.groupby(
        ["Spectrum ID", "Sequence", "Charge", "Modifications"]
    ):
        if grp["uCalc m/z"].nunique() == 1 and grp["Exp m/z"].nunique() == 1:
            continue
        if not already_warned:
            warnings.warn(
                "Recorded mass values are deviating across engines for identical predictions. Rerunning ursgal is recommended."
            )
            if click.confirm(
                "Would you like to continue with majority masses?", default=False
            ):
                pass
            else:
                raise ValueError(
                    "Recorded mass values are deviating across engines for identical predictions. Rerunning ursgal is recommended."
                )
            already_warned = True
        masses_calc = grp["uCalc m/z"].values
        masses_exp = grp["Exp m/z"].values
        majority_mass_calc = pd.value_counts(
            masses_calc, sort=True, ascending=False
        ).index.tolist()[0]
        new_df.loc[grp.index, "uCalc m/z"] = majority_mass_calc
        majority_mass_exp = pd.value_counts(
            masses_exp, sort=True, ascending=False
        ).index.tolist()[0]
        new_df.loc[grp.index, "Exp m/z"] = majority_mass_exp
        if (
            new_df.loc[grp.index, "uCalc m/z"].nunique() != 1
            or new_df.loc[grp.index, "Exp m/z"].nunique() != 1
        ):
            print("here")
    return new_df


def calc_deltas(grp, delta_lookup=None):
    for e in delta_lookup.keys():
        _3largest_values = grp[e].nlargest(3)
        # what if 0
        if _3largest_values.iloc[0] == 0:
            for delta_dict in delta_lookup[e]:
                grp[delta_dict["column"]] = np.nan
        else:
            for delta_dict in delta_lookup[e]:
                try:
                    grp[delta_dict["column"]] = (
                        grp[e] - _3largest_values.iloc[delta_dict["iloc"]]
                    )
                except IndexError:
                    pass
    return grp


def parse_protein_ids(
    protein_id,
):
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


def preprocess_df(df, features=None):
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
        labels=df[df.Sequence.str.contains("X") == True].index,
        axis=0,
        inplace=True,
    )

    # Missing mScores are replaced by minimum value for mScore in data
    if "mScore" in df.columns:
        min_mscore = df[df["mScore"] != 0]["mScore"].min()
        df.loc[df["mScore"] == 0, "mScore"] = min_mscore

    return df, features


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

        if engine == "omssa" and engine_df.Score.min() < 1e-30:
            stats[engine]["min_score"] = 1e-30
        else:
            stats[engine]["min_score"] = engine_df.Score.min()
        stats[engine]["max_score"] = engine_df.Score.max()
    return stats


def row_features(
    df, cleavage_site="C", proton=1.00727646677, max_charge=None, features=None
):
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
        lambda row: transform_score(row["Score"], stats[row["engine"]]), axis=1
    )
    features["row_features"].add("Score_processed")

    features["transformed_features"].add("Score")

    df = mass_sanity_check(df)
    df["Mass"] = (df["uCalc m/z"] - proton) * df["Charge"]
    features["row_features"].add("Mass")
    features["transformed_features"].add("uCalc m/z")

    df["dM"] = df["uCalc m/z"] - df["Exp m/z"]
    features["row_features"].add("dM")
    features["transformed_features"].add("Exp m/z")

    # Only works with trypsin for now
    if cleavage_site == "C":
        df["enzN"] = df.apply(
            lambda x: test_cleavage_aa(x["Sequence Pre AA"], x["Sequence Start"]),
            axis=1,
        )
        features["row_features"].add("enzN")

        df["enzC"] = df.apply(
            lambda x: test_sequence_aa_c(x["Sequence"][-1], x["Sequence Post AA"]),
            axis=1,
        )
        features["row_features"].add("enzC")

    else:
        raise ValueError("Only cleavage sites consistent with trypsin are accepted.")

    df["enzInt"] = df["Sequence"].str.count(r"[R|K]")
    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].apply(parse_protein_ids).apply(len)

    features["row_features"] |= set(["enzInt", "PepLen", "CountProt"])
    features["row_features"].add("Charge")

    return df, features


def col_features_alt(df, min_data=0.7, features=None):
    delta_lookup = {}
    for e, engine_grp in df.groupby("engine"):
        psm_counts_per_spec = engine_grp["Spectrum ID"].value_counts()
        specs_with_2_psms = psm_counts_per_spec[psm_counts_per_spec >= 2]
        specs_with_3_psms = psm_counts_per_spec[psm_counts_per_spec >= 3]
        if specs_with_2_psms.count() / psm_counts_per_spec.count() > min_data:
            delta_lookup[f"Score_processed_{e}"] = [
                {"column": f"delta_score_2_{e}", "iloc": 1}
            ]
            features["col_features_alt"].add(f"delta_score_2_{e}")
        if specs_with_3_psms.count() / psm_counts_per_spec.count() > min_data:
            delta_lookup[f"Score_processed_{e}"].append(
                {"column": f"delta_score_3_{e}", "iloc": 2}
            )
            features["col_features_alt"].add(f"delta_score_3_{e}")

    #
    core_id_cols = [
        # "Raw data location",
        "Spectrum Title",
        "Spectrum ID",
        "Sequence",
        "Modifications",
        "Is decoy",
        "Protein ID",
    ]
    value_cols = ["Score_processed"]
    features["transformed_features"].add("Score_processed")
    id_cols_full = []
    for c in df.columns:
        if c in core_id_cols:
            continue
        if c in value_cols:
            continue
        if c == "engine":
            continue
        id_cols_full.append(c)
    #
    number_of_entries = df.shape[0]
    for c in core_id_cols + id_cols_full:
        if df[c].count() != number_of_entries:
            # we have nones
            print("Filling nan in column ", c)
            df[c].fillna(0, inplace=True)
    cdf = pd.pivot_table(
        df,
        index=core_id_cols + id_cols_full,
        values=value_cols,
        columns="engine",
    )
    new_columns = []
    for l1, l2 in cdf.columns:
        if l2 == "":
            new_columns.append(l1)
        else:
            new_columns.append(f"{l1}_{l2}")
            features["col_features_alt"].add(f"{l1}_{l2}")
    cdf.columns = new_columns
    for c in cdf.columns:
        if c.startswith("Score_processed"):
            cdf[c].fillna(0, inplace=True)
    cdf.reset_index(inplace=True)

    # import numpy as np
    # cdf.to_csv("pre_delta_calculation.csv")
    print("Calculating delta_score columns...")
    for e in delta_lookup.keys():
        for delta_dict in delta_lookup[e]:
            cdf[delta_dict["column"]] = np.nan

    cdf = cdf.groupby("Spectrum ID").apply(calc_deltas, delta_lookup=delta_lookup)

    for c in cdf.columns:
        if c.startswith("delta_score"):
            cdf[c].fillna(cdf[c].min(), inplace=True)
    return cdf, features


def col_features(df, min_data=0.7, features=None):
    """
    Calculate col-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
    Returns:
        df (pd.DataFrame): input dataframe with added col-level features for each PSM
    """
    for e, engine_grp in df.groupby("engine"):
        psm_counts_per_spec = engine_grp["Spectrum ID"].value_counts()
        specs_with_2_psms = psm_counts_per_spec[psm_counts_per_spec >= 2]
        specs_with_3_psms = psm_counts_per_spec[psm_counts_per_spec >= 3]
        delta_lookup = []

        if specs_with_2_psms.count() / psm_counts_per_spec.count() > min_data:
            delta_lookup.append({"column": "delta_score_2", "iloc": 1})
            df.loc[engine_grp.index, "delta_score_2"] = np.nan
        if specs_with_3_psms.count() / psm_counts_per_spec.count() > min_data:
            delta_lookup.append({"column": "delta_score_3", "iloc": 2})
            df.loc[engine_grp.index, "delta_score_2"] = np.nan

        if len(delta_lookup) > 0:
            for spec_id, grp in engine_grp.groupby("Spectrum ID"):
                _3largest_values = grp["Score_processed"].nlargest(3)
                # Setting max distance by default

                for delta_dict in delta_lookup:
                    df.loc[grp.index, delta_dict["column"]] = _3largest_values.iloc[0]

                    try:
                        df.loc[grp.index, delta_dict["column"]] = (
                            grp["Score_processed"]
                            - _3largest_values.iloc[delta_dict["iloc"]]
                        )
                    except IndexError:
                        pass

    # log of the number of times the peptide sequence for a spectrum is found in the set
    df["lnNumPep"] = df.groupby("Sequence")["Sequence"].transform("count").apply(np.log)
    features["cols_features"] |= set(["lnNumPep", "delta_score_2", "delta_score_3"])
    return df, features


def determine_cols_to_be_dropped(df, features=None):
    """Determine what columns  need ot be dropped form the dataframe"""
    cols_to_drop = []
    all_feature_cols = set()
    for tag, feature_set in features.items():
        all_feature_cols |= feature_set
    all_feature_cols -= features["transformed_features"]

    for c in df.columns:
        if c not in all_feature_cols:
            cols_to_drop.append(c)
    return cols_to_drop


def calc_features(df, cleavage_site=None, old_cols=None, min_data=0.0, features=None):
    """
    Main function to calculate features from unified ursgal dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        cleavage_site (str): enzyme cleavage site (Currently only "C" implemented and tested)
        old_cols (List): columns initially in the dataframe
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
        features (dict):


    Returns:
        df (pd.DataFrame): input dataframe with added features for each PSM
    """
    pd.set_option("max_columns", 10000)
    print("Preprocessing df")
    df, features = preprocess_df(df, features)
    # print(features)
    # exit(1)
    print("Calculating row features")
    df, features = row_features(df, cleavage_site=cleavage_site, features=features)

    # df, features = col_features(df, min_data=min_data, features=features)
    cols_to_drop = determine_cols_to_be_dropped(df, features=features)
    df.drop(columns=cols_to_drop, inplace=True)
    # df.to_csv("test3.csv", index=False)
    print("Calculating col features")
    df, features = col_features_alt(df, features=features)
    # df.to_csv("test4.csv", index=False)
    return df, features
