from peptideForest import results


def plot_num_psms_by_method(
    df, methods, output_file,
):
    """
    Plot the number of PSMs for each method with available results.
    Args:
        df (pd.DataFrame): dataframe containing results from search engines and ML training
        methods (List): list of methods to use, if None, use all methods
        output_file (str): path to save new dataframe to
    """


def plot_num_psms_against_q(
    df_training, q_cut, methods, output_file,
):
    """
    Plot the number of PSMs for each method with available results.
    Args:
        df_training (pd.DataFrame): dataframe containing results from search engines and ML training
        q_cut (float): q-value used to identify top targets
        methods (List): list of methods to use, if None, use all methods
        output_file (str): path to save new dataframe to

    """


def plot_ranks(
    df, x_name, y_name, use_top_psm, n_psms, output_file, show_plot,
):
    """
    Plot ranks of PSMs from two methods against each other.
    Args:
        df (pd.DataFrame): dataframe containing ranks from different methods
        x_name (str): name of method for the x-axis
        y_name (str): name of method for the y-axis
        use_top_psm: use the top PSMs for each spectrum (True, default), otherwise use all PSMs
        n_psms: number of psms to plot results for, in multiples of the number with q-value<1%
        output_file: name of output file for image. If None (default), don't save the image

    """


def all(
    df_training, q_cut, methods, output_file,
):
    """
    Main function to plot.
    Args:
        df_training (pd.DataFrame): dataframe containing results from search engines and ML training
        q_cut (float): q-value used to identify top targets
        methods (List): list of methods to use, if None, use all methods
        output_file (str): path to save new dataframe to

    """
    # plot_num_psms_by_method
    # plot_num_psms_against_q
    # plot_ranks
