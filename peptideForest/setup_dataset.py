def combine_ursgal_csv_files(
        path_dict,
        output_file,
):
    """
    Combine separate ursgal search output csv files and return a dataframe. Also output as new csv file (optional).
    Takes a dictionary of input csv files and their respective engine and name of score column.

    Args:
        path_dict (str): path to ursgal path dict as .json
        output_file (str, optional): path to save new data frame to, do not save if None (default)
    Returns:
        input_df (pd.DataFrame): combined dataframes
    """


# defaults allow for reduction: input; combine_eng=True; keep_ursgal = false
def extract_features(
        input_df,
):
    """
    Calculate features from dataframe containing raw data from a single experiment.

    Args:
        input_df (pd.DataFrame): ursgal dataframe containing experiment data
    Returns:
        (Tuple): Tuple containing in respective order:
            output_df (pd.DataFrame): new dataframe containing the original experiment data and extracted features
            old_cols (List): columns initially in the dataframe
            feature_cols (List): column names of newly calculated features
    """
    # output_df = input_df
    # combine make_dataset and get_features
    # calls prep.calc_features
