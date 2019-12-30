from peptideForest import models


def get_shifted_psms(df, x_name, y_name, n_return):
    """
    Make dataframes showing which PSMs were top-targets before training but no longer are,
    and vice-versa.
    Args:
        df (pd.DataFrame): dataframe with training data and analysed results
        x_name (str): name of method used for baseline (e.g. search engine name)
        y_name (str): name of method used for comparison (e.g. ML model name)
        # [TRISTAN] fixed to false?
        n_return (int): number of examples to return. Returns the n_return most extreme examples,
                        i.e. those with the biggest jump in ranking. Default is 10

    Returns:
        df_new_top_targets (pd.DataFrame): dataframe containing information on the new top targets
        df_old_top_targets (pd.DataFrame): dataframe containing information on the new old targets

    """


def get_top_targets(
    df,
    # [TRISTAN] same here
    q_cut,
):
    """
    Identify which PSMs in a dataframe are top targets (i.e. is the highest ranked target PSM for a given spectrum
    with q-value < i%). Returns results for all Score_processed columns.
    Args:
        df (pd.DataFrame): dataframe with q-values for each PSM
        q_cut (float): q-value to use as a cut off as fraction

    Returns:
        df (pd.DataFrame): dataframe with new columns indicating if the PSM is a top target

    """


def calc_final_q_vals(
    df, col, frac_tp, top_psm_only, initial_engine,
):
    """
    Calculate q-value for given score column.
    Args:
        df (pd.DataFrame): dataframe with training results
        col (str): name of method to calculate q-values for
        frac_tp (float): estimate of fraction of true positives in target dataset
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        initial_engine (str):   method which was used to originally rank the PSMs, to be used here as the second ranking
                                column

    Returns:
        df (pd.DataFrame): same dataframe with q-values added as column

    """
    # models.get_q_values


def get_ranks(
    df,
    # [TRISTAN] very much removable? fixed "first"
    methods,
    from_scores,
):
    """
    Add a column with the rank of each PSM for all Score_processed columns.
    Args:
        df (pd.DataFrame): dataframe with scores for each PSM
        methods (str):  method to use (based on pandas rank method). Default is first,
                        such that PSMs with the same score are ranked on the order they appear in the dataframe
        from_scores (bool): rank by scores if True, rank by q-values if False

    Returns:
        df (pd.DataFrame): same dataframe with new columns indicating the ranks

    """


def calc_all_final_q_vals(
    df, frac_tp, top_psm_only, initial_engine,
):
    """
    Calculate the q-values for all score columns
    Args:
        df (pd.DataFrame): dataframe with training results
        frac_tp (float): estimate of fraction of true positives in target dataset
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        initial_engine (str):   method which was used to originally rank the PSMs, to be used here as the second ranking
                                column

    Returns:
        df (pd.DataFrame): input dataframe with q-values added as new columns

    """


def analyse(
    df_training,
    initial_engine,
    # [TRISTAN] ist das immer dasselbe wie q_val_cut was da eigentlich stehen sollte? vorher immer fix auf 0.01; from_scores ist auch == True fix
    q_cut,
    from_scores,
):
    """
    Main function to analyse results.
    Args:
        df_training (pd.DataFrame): dataframe with q-values for each PSM
        initial_engine (str):
        q_cut (float): q-value to use as cut off

    Returns:

    """
    # blabla aus 1_0_0
    # calc_all_final_q_vals(df_training, from_method=initial_engine)
    # get_top_targets
    # get_ranks
    # get_shifted_psms
