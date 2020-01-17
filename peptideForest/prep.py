import re
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
AA_DELIM_REGEX = re.compile(r"<\|>|;")
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
    all_aas = set(re.split(AA_DELIM_REGEX, aa_field))
    return any(aa in CLEAVAGE_SITES for aa in all_aas) or aa_start in ["1", "2"]


def test_sequence_aa_c(
    aa, pre_post_aa,
):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
    Args:
        aa (str): start/end of sequence
        pre_post_aa (str): (multiple) pre/post amino acids
    Returns:
        (bool): True if start/end is consistent with cleavage site, False otherwise
    """
    return aa in CLEAVAGE_SITES or "-" in pre_post_aa


# only if cleavage site included
def test_sequence_aa_n(
    aa, aa_start,
):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
    Args:
        aa (str): start/end of sequence
        aa_start (str): start of sequence
    Returns:
        (bool): True if start/end is consistent with cleavage site, False otherwise
    """
    return aa in CLEAVAGE_SITES or aa_start in ["1", "2"]


def parse_protein_ids(protein_id,):
    """
    Turns ursgal dataframe column "Protein ID" into a list of all protein IDs.
    Args:
        protein_id (str): separated ProteinIDs
    Returns:
        prot_id_set (: True if start/end is consistent with cleavage site, False otherwise
    """
    # [TRISTAN]
    # sep = "<|>" hier drinnen, since provided by ursgal anyways? -> Doch und Decoy in config.json oder so
    sep = "<|>"
    clean = protein_id.replace("decoy_", "").strip()
    prot_id_set = list(clean.split(sep))
    return prot_id_set


def transform_score(score, engine, minimum_score):
    """
    Transforms a score to a base 10 logarithmic range.
    Args:
        score (float): score value from engine
        engine (str): database search engine that generated score
        minimum_score (float): used when score is negative or 0
    Returns:
        score (float): transformed score
    """
    if "_" in engine:
        eng = engine.split("_")[0]
    else:
        eng = engine

    if eng not in ENGINES:
        raise ValueError(f"Engine {engine} not known")

    elif ENGINES[eng][1]:

        if score > 0:
            if score >= 1e-30:
                transformed_score = -np.log10(score)
            else:
                # score can get very small, set to -log10(1e-30) if less than 1e-30
                transformed_score = 30.0
        else:
            transformed_score = minimum_score

        return transformed_score

    return score


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

    # Initialize to nan (for PSMs from different engines)
    df[col] = np.nan
    # [TRISTAN] what is meant?:^--- this delta should be be on engine level -> extra parameter

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
            ith_best = ith_best["Score_processed"].transform(
                lambda x: x.nlargest(i).min()
            )
            df.loc[inds, col] = df.loc[inds, "Score_processed"] - ith_best
            mean_val = df.loc[inds, col].mean()
            # Replace missing with mean
            inds = df_engine.loc[
                df_engine["Spectrum ID"].isin(psm_counts[psm_counts < i].index), :
            ].index
            df.loc[inds, col] = mean_val
    return df


def get_top_targets_decoys(df,):
    """
    Get the top target and top decoy for each Spectrum ID based on score for each engine.
    # [TRISTAN] do we want balance_dataset??
    Args:
        df (pd.DataFrame): ursgal dataframe
    Returns:
        df (pd.DataFrame): ursgal dataframe with only top targets/decoys
    """
    dfs_engine = {
        engine: df[df["engine"] == engine] for engine in df["engine"].unique()
    }
    df = None

    for df_engine in dfs_engine.values():

        # Get top targets
        targets = df_engine[~df_engine["Is decoy"]]
        targets = targets.sort_values(
            "Score_processed", ascending=False
        ).drop_duplicates(["Spectrum ID"])

        # Get top decoys
        decoys = df_engine[df_engine["Is decoy"]]
        decoys = decoys.sort_values("Score_processed", ascending=False).drop_duplicates(
            ["Spectrum ID"]
        )

        # Merge them together
        df_engine = pd.concat([targets, decoys]).sort_index()

        # [TRISTAN] see above # if balance_dataset: --> Kann weg
        # also see commented line below which is redundant?
        # if balance_dataset:
        #     # Only keep those where we have one target and one decoy
        #     # spec_id_counts = df_engine[df_engine["Spectrum ID"].value_counts() == 2]
        #     spec_id_counts = df_engine["Spectrum ID"].value_counts()
        #     spec_id_counts = spec_id_counts[spec_id_counts == 2]
        #     df_engine = df_engine.loc[
        #         df_engine["Spectrum ID"].isin(spec_id_counts.index), :
        #     ]

        if df:
            df = pd.concat([df, df_engine.copy(deep=True)]).sort_index()
        else:
            df = df_engine.copy(deep=True)

    return df


def preprocess_df(df,):
    """
    Preprocess ursgal dataframe:
    # [TRISTAN]: Does it though? Commented out? Map amino acid isomers to single value (I); --> Kann weg aber in docstring -> sanity check
    Remove decoy PSMs overlapping with targets and fill missing modifications (None).
    Sequences containing 'X' are removed.
    Operations are performed inplace on dataframe!
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Returns:
        df (pd.DataFrame): preprocessed dataframe
    """
    # df['Sequence'] = df['Sequence'].str.replace('L', 'I')

    # [TRISTAN] critical change but as discussed is probably correct
    decoys = df[df["Is decoy"]]
    targets = df[~df["Is decoy"]]

    # Overlaps in Sequences between targets and decoys are removed -> Warning droppen [TRISTAN]
    df.drop(
        decoys[decoys["Sequence"].isin(targets["Sequence"].unique())].index,
        inplace=True,
    )

    # Fill missing modifications
    df["Modifications"].fillna("None", inplace=True)

    # Remove Sequences with "X"
    df.drop(
        labels=df[df.Sequence.str.contains("X") == True].index, axis=0, inplace=True
    )

    return df


def get_stats(df):
    """
    Calculate minimum scores across all PSMs for all engines.
    Args:
        df (pd.DataFrame): ursgal dataframe
    Returns:
        (Dict of str: Any): Dict of engines containing dict of min scores.
    """
    stats = {}
    engines = df["engine"].unique()

    for engine in engines:
        stats[engine] = {}
        engine_df = df[df["engine"] == engine]

        # Minimum score (transformed)
        stats[engine]["min_score"] = -np.log10(
            engine_df.loc[engine_df["Score"] > 1e-30, "Score"].min()
        )

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
            "Mass",
            "delta m/z",
            "abs delta m/z",
            "ln abs delta m/z",
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
        eng_names = engine.split("_")[0]
        # Rename the columns that will have different names
        cols_single_engine = [f"{c}_{eng_names}" for c in cols_single]
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
    df_combine = df_combine.drop(
        [c for c in df_combine.columns if len(df_combine[c].unique()) == 1], axis=1
    )

    # Drop rows that are identical
    df_combine = df_combine.drop_duplicates()

    # Average mass based columns and drop the engine specific ones
    for col in ["Mass", "delta m/z", "abs delta m/z", "ln abs delta m/z"]:
        eng_names = [engine.split("_")[0] for engine in df["engine"].unique()]
        cols = [f"{col}_{eng_name}" for eng_name in eng_names]
        df_combine[col] = df_combine[cols].mean(axis=1)
        df_combine = df_combine.drop(cols, axis=1)

    return df_combine


def row_features(df, cleavage_site):
    """
    Calculate row-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        cleavage_site (str): enzyme cleavage site
    Returns:
        df (pd.DataFrame): input dataframe with added row-level features for each PSM
    """
    stats = get_stats(df)

    # Calculate processed score
    df["Score_processed"] = df.apply(
        lambda row: transform_score(
            row["Score"], row["engine"], stats[row["engine"]]["min_score"]
        ),
        axis=1,
    )

    df["Mass"] = (df["Exp m/z"] * df["Charge"]) - (df["Charge"] - 1) * PROTON
    df["delta m/z"] = df["uCalc m/z"] - df["Exp m/z"]
    df["abs delta m/z"] = df["delta m/z"].apply(np.absolute)

    # Get the log of delta mass and replace values that give log(0) with minimum
    log_min = np.log(df.loc[df["abs delta m/z"] > 0, "abs delta m/z"].min())
    df["ln abs delta m/z"] = df["abs delta m/z"].apply(
        lambda x: np.log(x) if x != 0 else log_min
    )

    # [TRISTAN] Add columns indicating cleavage site consistency -> only works with trypsin for now add comment / hier muss noch exception else dazu
    if cleavage_site == "C":
        df["enzN"] = df.apply(
            lambda x: test_cleavage_aa(x["Sequence Pre AA"], x["Sequence Start"]),
            axis=1,
        )
        df["enzC"] = df.apply(
            lambda x: test_sequence_aa_c(x["Sequence"][-1], x["Sequence Post AA"]),
            axis=1,
        )

    df["enzInt"] = df.apply(
        lambda row: sum(1 for aa in row["Sequence"] if aa in CLEAVAGE_SITES), axis=1
    )
    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].apply(parse_protein_ids).apply(len)

    # Get maximum charge to use for columns
    max_charge = df["Charge"].max(axis=0)

    # Create categorical charge columns
    for i in range(max_charge):
        df[f"Charge{i+1}"] = df["Charge"] == i + 1
    df[f">Charge{max_charge+1}"] = df["Charge"] > max_charge

    return df


def col_features(df,):
    """
    Calculate col-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Returns:
        df (pd.DataFrame): input dataframe with added col-level features for each PSM
    """
    # [TRISTAN] make min_data an argument?
    # delta_score_2: difference between first and second score for a spectrum.
    # If < min_data have two scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=2, min_data=0.7)

    # delta_score_3: difference between first and third score for a spectrum.
    # If < min_data have three scores for both target and decoys, don't calculate.
    df = calc_delta_score_i(df, i=3, min_data=0.7)

    # log of the number of times the peptide sequence for a spectrum is found in the set
    df["lnNumPep"] = df.groupby("Sequence")["Sequence"].transform("count").apply(np.log)

    return df


def calc_features(
    df, old_cols,
):
    """
    Main function to calculate features from unified ursgal dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
        old_cols (List): columns initially in the dataframe
    Returns:
        df (pd.DataFrame): input dataframe with added features for each PSM
    """
    df = preprocess_df(df)
    df = row_features(df, cleavage_site="C")
    df = col_features(df)

    feature_cols = list(set(df.columns) - set(old_cols))
    df = combine_engine_data(df, feature_cols)

    return df

    # [TRISTAN] cleavage_site to be included? -> erledigt
