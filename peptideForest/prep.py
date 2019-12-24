def test_cleavage_aa(
        aa_field,
        aa_start,
):
    """
    Test whether pre/post amino acid is consistent with enzyme cleavage site.
    Args:
        aa_field (str): (multiple) pre/post amino acids
        aa_start (str): start of sequence
    Return:
        (bool): True if amino acid is consistent with cleavage site, False otherwise
    """


def test_sequence_aa_c(
        aa,
        pre_post_aa,
):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
    Args:
        aa (str): start/end of sequence
        pre_post_aa (str): (multiple) pre/post amino acids
    Return:
        (bool): True if start/end is consistent with cleavage site, False otherwise
    """


# only if cleavage site included
def test_sequence_aa_n(
        aa,
        pre_post_aa,
):
    """
    Test whether start/end of sequence is consistent with enzyme cleavage site, or if it is cut at end.
    Args:
        aa (str): start/end of sequence
        pre_post_aa (str): (multiple) pre/post amino acids
    Return:
        (bool): True if start/end is consistent with cleavage site, False otherwise
    """


def parse_protein_ids(
        protein_id,
):
    """
    Turns ursgal dataframe column "Protein ID" into a list of all protein IDs.
    Args:
        protein_id (str): separated ProteinIDs
    Return:
        prot_id_set (: True if start/end is consistent with cleavage site, False otherwise
    """
    # sep = "<|>" hier drinnen, since provided by ursgal anyways?
    # was war das mit Zeile 87 again? die sollte raus?? clean = protein_id.replace("decoy_", "").strip()


def calc_delta_score_i(
        df,
        i,
        min_data,
):
    """
    Calculate delta_score_i, which is the difference in score between a PSM and the ith ranked PSM for a given
    spectrum. It is calculated for targets and decoys combined. It is only calculated when the fraction of
    spectra with more than i PSMs  is greater than min_data. Missing values are replaced by the mean.
    It is calculated for each engine.
    Args:
        df (pd.DataFrame): ursgal dataframe
        i (int): rank to compare to (i.e. i=2 -> subtract score of 2nd ranked PSM)
        # [TRISTAN] do we want min_data??
        min_data (float): minimum fraction of spectra for which we require that there are at least i PSMs
    Return:
        df (pd.DataFrame): ursgal dataframe with delta_score_i added
    """


def get_top_targets_decoys(
        df,
):
    """
    Get the top target and top decoy for each Spectrum ID based on score for each engine.
    # [TRISTAN] do we want balance_dataset??
    Args:
        df (pd.DataFrame): ursgal dataframe
    Return:
        df (pd.DataFrame): ursgal dataframe with only top targets/decoys
    """


def preprocess_df(
        df,
):
    """
    Preprocess ursgal dataframe:
    [TRISTAN]: Does it though? Commented out? Map amino acid isomers to single value (I);
    Remove decoy PSMs overlapping with targets and fill missing modifications (None).
    Sequences containing 'X' are removed.
    Operations are performed inplace on dataframe!
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Return:
        df (pd.DataFrame): preprocessed dataframe
    """


def get_stats(
        df,
):
    """
    Calculate minimum scores across all PSMs for all engines.
    Args:
        df (pd.DataFrame): ursgal dataframe
    Return:
        (Dict of str: Any): Dict of engines containing dict of min scores.
    """


def combine_engine_data(
    df,
    feature_cols,
):
    """
    Calculate row-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): dataframe containing search engine results
        feature_cols (list): list of calculated feature columns in dataframe
    Return:
        df (pd.DataFrame):  dataframe with results for each search engine combined for each individual
                            experimental-theoretical PSM.
    """


def row_features(
        df,
):
    """
    Calculate row-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Return:
        df (pd.DataFrame): input dataframe with added row-level features for each PSM
    """
    # get_stats now called from within here

    # test_cleavage_aa
    # test_sequence_aa_c

    # test_cleavage_aa
    # test_cleavage_aa_n

    # parse_protein_ids


def col_features(
        df,
):
    """
    Calculate col-level features from unified ursgal dataframe.
    Features are added as columns inplace in dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Return:
        df (pd.DataFrame): input dataframe with added col-level features for each PSM
    """
    # calc_delta_score_i
    # get_top_target_decoys


def calc_features(
        df,
):
    """
    Main function to calculate features from unified ursgal dataframe.
    Args:
        df (pd.DataFrame): ursgal dataframe containing experiment data
    Return:
        df (pd.DataFrame): input dataframe with added features for each PSM
    """
    # preprocess_df()
    # row_features()
    # col_features()
    # 524-525 -> combine_engine_data

    # [TRISTAN] cleavage_site to be included?
