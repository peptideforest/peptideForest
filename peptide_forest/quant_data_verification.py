"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Make a dataset from target-decoy data to be used by percolator iterative method
"""

# pylint: disable=unused-import
from typing import Any, Dict, Pattern, Set, Tuple, List, Union
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from .models import ml_methods
from .analyse_results import get_top_targets


def read_quant_data(file_name: str) -> pd.DataFrame:
    """
  Read in a csv file containing quant data from a single experiment
  Arguments:
    - file_name: name of csv file containing quant results
  Returns:
    - df: dataframe containing original data and features
  """
    df_quant = pd.read_csv(file_name, index_col=0)
    df_quant["SPECTRUM_ID"] = df_quant["SPECTRUM_ID"].apply(
        lambda x: int(str(x).split("F")[-1])
    )
    cols = ["SPECTRUM_ID", "MAXAREA", "ISOTOPELABEL_ID", "S2I"]
    return df_quant[cols]


def get_maxarea_per_spectrum(
    df: pd.DataFrame, file_name: str, s2i_cut: float = -1.0
) -> pd.DataFrame:
    """
  Get the maxarea values for each peptide for each protein
  Arguments:
    - df: dataframe with score results
    - file_name: name of csv file containing quant results
    - s2i_cut: cut-off for signal to interference, default is -1 (don't cut-off)
  Returns:
    - df: input dataframe with maxarea for each channel added
  """
    # read the quant data
    df_quant = read_quant_data(file_name)

    # filter by S2I score (signal to interference)
    df_quant = df_quant[df_quant["S2I"] > s2i_cut]

    # pivot the table to get maximum maxarea for each sequence and rename the columns
    df_quant = pd.pivot_table(
        df_quant,
        values=f"MAXAREA",
        index=["SPECTRUM_ID"],
        columns=["ISOTOPELABEL_ID"],
        aggfunc=np.max,
    )
    df_quant = df_quant.reset_index()
    dict_change_names = {i: f"channel_{i}" for i in range(62, 72)}
    df_quant = df_quant.rename(dict_change_names, axis=1)
    df_quant = df_quant.rename({"SPECTRUM_ID": "Spectrum ID"}, axis=1)

    # get the names of the new maxarea columns
    cols = [f"channel_{i}" for i in range(62, 72)]

    # merge results and quant data, and fill NaNs with 0
    df = df.merge(df_quant[["Spectrum ID"] + cols], on="Spectrum ID", how="left")
    df[cols] = df[cols].fillna(0)

    return df


def get_ratios_method(
    df: pd.DataFrame,
    i: int,
    j: int,
    method: Union[str, Tuple[str, str]] = None,
    log2: bool = False,
    drop_no_data: bool = True,
) -> pd.DataFrame:
    """
  Calculate the ratios between channel i and channel j for all top-targets
  identified by method. Also return the sum of the channels
  Arguments:
    - df: dataframe containing results and channel values
    - method: name of method to use for top-targets. If a string is provided,
              all top-targets for that method are used. If a tuple (of length 2)
              is provided, top-targets that are identified by the first method
              in the list but not by the secons=d are used. If method = None (default)
              then ratios and sums are returned for all PSMs that are a top-target
              for at least one method. If None also returns columns for 'all-engines' and
              'any-engine'
    - i, j: channels to calcualte ratio for (i/j)
    - log2: calculate log2 of the ratio, default = True
    - drop_no_data: drop spectra where at least one of the channels is equal to 0,
                    default is False
  Returns:
    - df_ratios: dataframe containing ratio i/j and sum i+j, with column indicating if
                 the protein is human or ecoli for that PSM
  """

    # list of search engines
    engines = [e for e in df.columns if "top_target_" in e]
    engines = [
        e for e in engines if all(ml_method not in e for ml_method in ml_methods)
    ]

    # select the appropriate PSMs based on method
    if isinstance(method, str):
        df_ratios = df[df[f"top_target_{method}"]].copy(deep=True)
        df_ratios = df_ratios[[f"channel_{i}", f"channel_{j}"]]
    elif isinstance(method, list):
        df_ratios = df[
            df[f"top_target_{method[0]}"] & ~df[f"top_target_{method[1]}"]
        ].copy(deep=True)
        df_ratios = df_ratios[[f"channel_{i}", f"channel_{j}"]]
    else:
        cols = [c for c in df.columns if "top_target" in c]
        df_ratios = df[df[cols].any(axis=1)].copy(deep=True)
        df_ratios = df_ratios[cols + [f"channel_{i}", f"channel_{j}"]]
        df_ratios["top_target_all-engines"] = df_ratios[engines].all(axis=1)
        n_majority_engines = np.ceil(0.5 * len(engines))
        df_ratios["top_target_majority-engines"] = (
            df_ratios[engines].sum(axis=1) >= n_majority_engines
        )
        df_ratios["top_target_any-engine"] = df_ratios[engines].any(axis=1)
    # make the Is human column, True if protein for a PSM is human, false if E-coli
    df_ratios["Is human"] = (
        df.loc[df_ratios.index, "Protein ID"].str.lower().str.contains("human")
    )
    # remove any PSMs where are least 1 channel = 0
    if drop_no_data:
        df_ratios = df_ratios[
            (df_ratios[[f"channel_{i}", f"channel_{j}"]] != 0).all(axis=1)
        ]
    # get the ratios and sums
    df_ratios[f"ratio {i}/{j}"] = df_ratios[f"channel_{i}"] / df_ratios[f"channel_{j}"]
    df_ratios[f"sum {i}+{j}"] = df_ratios[f"channel_{i}"] + df_ratios[f"channel_{j}"]
    # take log_2 of the ratio
    if log2:
        df_ratios[f"log2 (ratio {i}/{j})"] = np.log2(df_ratios[f"ratio {i}/{j}"])
    # remove the channel values
    df_ratios = df_ratios.drop([f"channel_{i}", f"channel_{j}"], axis=1)
    return df_ratios


def get_slice_mean_error(
    df_input: pd.DataFrame,
    n_slices: int = 10,
    is_ecoli: bool = True,
    quant_cut: bool = True,
    bins_in: np.array = None,
    bin_inv_sum: bool = False,
    bin_log_sum: bool = False,
    base: int = 2,
    mean_of_log: bool = False,
) -> pd.DataFrame:
    """
  Calculate the mean and error of the ratio in slices of sum values.
  Arguments:
    - df_input: DataFrame containing ratios and sums for two channels. All Nans must be
                removed first
    - n_slices: number of n_slices to use, default is 10
    - is_ecoli: use e-coli proteins, otherwise use human, default = True
    - quant_cut: use pd.qcut, taking quantiles instead of evenly spaced bins (default=True)
    - bins_in: bins to use. If not sent, will generate bins based on quant_cut (default=None)
    - bin_inv_sum: use the inverse sum to bin the data, default = False
    - bin_log_sum: use the log of the sum/inverse sum to bin the data, default = False
    - base: base to use for the logarithm
    - mean_of_log: take the log of the mean (False, default) or the mean of the logs (True)
  Returns:
    - df_mean: dataframe containing mean and standard deviations for each slice
    - bins_out: bins used for the calculations
  """
    # get either human or e-coli proteins
    if is_ecoli:
        df = df_input[~df_input["Is human"]].copy(deep=True)
    else:
        df = df_input[df_input["Is human"]].copy(deep=True)
    # get the name of the sum and ratio columns
    col_sum = df.filter(like="sum").columns[0]
    if bin_inv_sum:
        df[f"1 / {col_sum}"] = 1.0 / df[col_sum]
        col_sum = f"1 / {col_sum}"
        print(col_sum)
        print(df[col_sum].head())
        if bin_log_sum:
            df[f"log({col_sum})"] = -np.log10(df[col_sum])
            col_sum = f"log({col_sum})"
    else:
        if bin_log_sum:
            df[f"log({col_sum})"] = np.log10(df[col_sum])
            col_sum = f"log({col_sum})"
    col_ratio = df.filter(like="ratio").columns[0]

    # test if there are NaNs or infs in the ratio column
    if sum(df_input[col_ratio].isnull()) > 0:
        print("NaNs in ratio column")
        return None
    if sum(df_input[col_ratio] == np.inf) > 0:
        print("infs in ratio column")
        return None
    if bins_in is not None:
        # if bins were supplied, use these
        df.loc[:, "mid"] = (
            pd.cut(df[col_sum], bins=bins_in).apply(lambda x: x.mid).values
        )
        df.loc[:, "bin"] = pd.cut(df[col_sum], bins=bins_in, labels=False)
        bins_out = bins_in
    else:
        # otherwise use a given number of n_slices, either in quantiles or even sized bins
        if quant_cut:
            df.loc[:, "mid"] = (
                pd.qcut(df[col_sum], n_slices).apply(lambda x: x.mid).values
            )
            df.loc[:, "bin"], bins_out = pd.qcut(
                df[col_sum], n_slices, labels=False, retbins=True
            )
        else:
            df.loc[:, "mid"] = (
                pd.cut(df[col_sum].dropna(), n_slices).apply(lambda x: x.mid).values
            )
            df.loc[:, "bin"], bins_out = pd.cut(
                df[col_sum], n_slices, labels=False, retbins=True
            )
    df_bins = (
        df[["bin", "mid"]]
        .drop_duplicates()
        .dropna()
        .sort_values("mid")
        .reset_index(drop=True)
    )
    if mean_of_log:
        df.loc[:, f"log{base}({col_ratio})"] = df[col_ratio].apply(
            lambda x: np.log(x) / np.log(base)
        )
        col_ratio = f"log{base}({col_ratio})"
    df_means = (
        df.groupby("bin")[col_ratio]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("bin")
    )

    df_means.loc[:, "mid"] = df_bins["mid"].values.tolist()
    df_means.loc[:, "std"] = df_means["std"] / np.sqrt(df_means["count"])
    df_means = df_means.rename(
        {"mid": col_sum, "mean": f"<{col_ratio}>", "std": "error"}, axis=1
    )
    return df_means, bins_out


def get_logn_mean_error(
    df_input: pd.DataFrame,
    n_slices: int = 10,
    is_ecoli: bool = True,
    quant_cut: bool = True,
    bins_in: np.array = None,
    base: int = 2,
    bin_inv_sum: bool = False,
    bin_log_sum: bool = False,
    mean_of_log: bool = False,
) -> pd.DataFrame:
    """
  Calculate the log_n of the mean and error of the ratio in slices of sum values.
  Arguments:
    - df_input: DataFrame containing ratios and sums for two channels. All Nans must be
                removed first
    - n_slices: number of n_slices to use, default is 10
    - is_ecoli: use e-coli proteins, otherwise use human, default = True
    - quant_cut: use pd.qcut, taking quantiles instead of evenly spaced bins (default=True)
    - bins_in: bins to use. If not sent, will generate bins based on quant_cut (default=None)
    - base: base to use for the logarithm
    - bin_inv_sum: use the inverse sum to bin the data, default = False
    - bin_log_sum: use the log of the sum/inverse sum to bin the data, default = False
    - mean_of_log: take the log of the mean (False, default) or the mean of the logs (True)
  Returns:
    - df_mean: dataframe containing log_n of the mean and standard deviations for each slice
    - bins_out: bins used for the calculations
  """
    df_means, bins_out = get_slice_mean_error(
        df_input,
        n_slices=n_slices,
        is_ecoli=is_ecoli,
        quant_cut=quant_cut,
        bins_in=bins_in,
        bin_inv_sum=bin_inv_sum,
        bin_log_sum=bin_log_sum,
        base=base,
        mean_of_log=mean_of_log,
    )

    # get the name of the sum and ratio columns
    col_ratio = df_means.filter(like="ratio").columns[0]
    if mean_of_log:
        df_means = df_means.rename({"error": f"error [{col_ratio}]"}, axis=1)
    else:
        df_means.loc[:, "error"] = df_means["error"] / (
            df_means[col_ratio] * np.log(base)
        )
        df_means[col_ratio] = np.log(df_means[col_ratio]) / np.log(base)
        df_means = df_means.rename(
            {
                col_ratio: f"log{base}({col_ratio})",
                "error": f"error [log{base}({col_ratio})]",
            },
            axis=1,
        )

    return df_means, bins_out


def tanh_func(x: np.array, a: float, b: float, c: float) -> np.array:
    """
  Function to fit to channel ratio data
  Function is fitted to logn(<ratio>) against sum
  Arguments:
    - x: sum values
    - a, b, c: constants for the function
  Returns:
    - a * tanh(b * x / 5e7) + c
  """
    return a * np.tanh(np.absolute(b * x / 1e7)) + c


def atan_func(x: np.array, a: float, b: float, c: float) -> np.array:
    """
  Function to fit to channel ratio data
  Function is fitted to logn(<ratio>) against sum
  Arguments:
    - x: sum values
    - a, b, c: constants for the function
  Returns:
    - a * atan(b * x / 5e7) / (pi/2) + c
  """
    return a * np.arctan(np.absolute(b * x / 1e8)) / (0.5 * np.pi) + c


def log_func(x: np.array, a: float, b: float, c: float) -> np.array:
    """
  Function to fit to channel ratio data
  Function is fitted to logn(<ratio>) against sum
  Arguments:
    - x: sum_values
    - a, b, c: constants for the function
  Returns:
    -  a * exp(b * 1e7 / x) + c
  """
    return a * np.exp(np.absolute(b * 1e7 / x)) + c


def fit_function(
    df_means: pd.DataFrame, consts_guess: List = None, func_to_fit: str = "arctan"
) -> Tuple[float, float, List]:
    """
  Fit function func to data in df_means
  Arguments:
    - df_means: dataframe containing ratio, log of average mean and error
    - consts_guess: list containing initial guesses for constants in func
    - func_to_fit: function to fit to the data
                   - 'arctan': inverse tan function (default), a*arctan(b*x)+c
                   - 'log': log function, a*loc(b/x)+c
                   - 'tanh': tanh function, a*tanh(b*x)+c
  Returns:
    - mean_ratio_0: value of logn(<ratio>) at sum->infinity
    - error: error in the above
  """
    # get column names
    y_col = df_means.filter(like="ratio").columns[0]
    yerr_col = df_means.filter(like="error").columns[0]
    x_col = df_means.filter(like="sum").columns[0]

    # get data to fit to
    if "log" in x_col:
        df_means[x_col] = 10.0 ** df_means[x_col]
    if "inv" in x_col:
        df_means[x_col] = 1.0 / df_means[x_col]
    x = df_means[x_col].values
    y = df_means[y_col].values
    yerr = df_means[yerr_col].values

    # fit the data
    if func_to_fit == "log":
        func = log_func
    elif func_to_fit == "tanh":
        func = tanh_func
    else:
        func = atan_func

    # loop over all datapoints, skip one at a time, fit to the data for each dataset
    # leave the first and last point always in the set
    popt_all = []
    # for i_skip in np.arange(1, len(x) - 1):
    for i_skip in np.arange(1, len(x)):
        x = [x_sub for ii, x_sub in enumerate(x) if ii != i_skip]
        y = [y_sub for ii, y_sub in enumerate(y) if ii != i_skip]
        yerr = [yerr_sub for ii, yerr_sub in enumerate(yerr) if ii != i_skip]
        popt = curve_fit(func, x, y, p0=consts_guess, sigma=yerr, maxfev=5000)[0]
        popt_all.append([np.absolute(p) for p in popt])

    # get the value and the error
    popt = np.mean(popt_all, axis=0)
    mean_ratio_0 = popt[0] + popt[2]  # np.mean(y)#func(1e9, popt[0], popt[1], popt[2])#
    error = 2.0 * np.std([p[0] + p[2] for p in popt_all])  # np.std(y)/np.sqrt(len(y))#
    return mean_ratio_0, error, popt


def get_maxarea_per_protein(
    df: pd.DataFrame, target_col: str, s2i_cut: float = None
) -> pd.DataFrame:
    """
  Get the maxarea values for each peptide for each protein
  Arguments:
    - df: dataframe with quant and score results
    - target_col: top_target column, denoting which scoring method to get max_area for
    - s2i_cut: cut-off for signal to interference, default is None (don't cut-off)
  Returns:
    - df_quant_results: dataframe containing maxarea values for each peptide for each protein
  """
    # columns to group by
    group_cols = ["protein", "ISOTOPELABEL_ID", "Sequence"]
    # get top targets, from column target_col
    df_quant_results = df[df[target_col]]
    if s2i_cut:
        df_quant_results = df_quant_results[df_quant_results["S2I"] >= s2i_cut]
    # group results by group_cols and take maximum MAXAREA for that group
    df_quant_results = df_quant_results[(df_quant_results["ISOTOPELABEL_ID"] > 0)]
    df_quant_results = df_quant_results.groupby(group_cols)["MAXAREA"].max()
    # reset the index to make index into columns
    df_quant_results = df_quant_results.reset_index()
    return df_quant_results


def merge_results_quant(file_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
  Merge quant data from file_name with training results in dataframe df
  Arguments:
    - file_name: filename of the csv file containing the quant data (maxarea results)
    - df: dataframe containing score results (from search engines and ML methods)
  Returns:
    - df_quant_training: dataframe with score results merged with wuant data
  """

    # get the quant data
    df_quant = read_quant_data(file_name)

    # merge the training resilts with the quant data, on spectrum ID
    df_quant_training = df_quant.merge(
        df, left_on=["SPECTRUM_ID"], right_on=["Spectrum ID"]
    )

    # get top targets only
    methods = [c for c in df_quant_training.columns if "top_target_" in c]

    df_quant_training = df_quant_training[df_quant_training[methods].any(axis=1)]

    # and also drop all decoys (should be done by the above)
    df_quant_training = df_quant_training[~df_quant_training["Is decoy"]]
    # drop any duplicates from after the merge
    df_quant_training = df_quant_training.drop_duplicates()

    cols = df_quant_training.columns
    # set protein ID to be the index
    df_quant_training = (
        df_quant_training["Protein ID"]
        .apply(pd.Series)
        .merge(df_quant_training, right_index=True, left_index=True)
    )

    # melt the dataframe, to get data for each protein
    df_quant_training = df_quant_training.melt(id_vars=cols, value_name="protein")
    df_quant_training = df_quant_training.drop("variable", axis=1).dropna(
        subset=["protein"]
    )
    # columns to keep
    cols = [
        "ISOTOPELABEL_ID",
        "protein",
        "Spectrum ID",
        "Sequence",
        "MAXAREA",
        "S2I",
    ] + methods
    df_quant_training = df_quant_training[cols]

    df_quant_training["sum_top_targets"] = df_quant_training[methods].sum(axis=1)
    df_quant_training = df_quant_training.sort_values(
        ["protein", "Sequence", "ISOTOPELABEL_ID", "sum_top_targets"], ascending=False
    )
    df_quant_training = df_quant_training.drop_duplicates(
        ["protein", "Sequence", "ISOTOPELABEL_ID"]
    )
    return df_quant_training


def get_results_protein(file_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
  Get maxarea values for each peptide
  Arguments:
    - file_name: filename of the csv file containing the quant data (maxarea results)
    - df: dataframe containing score results (from search engines and ML methods)
  Results:
    - df_quant_results: dataframe containing score results and maxarea values
  """
    df_quant_training = merge_results_quant(file_name, df)

    # get a list of all methods
    methods = [c for c in df_quant_training.columns if "top_target_" in c]
    methods = [m.split("top_target_")[-1] for m in methods]

    group_cols = ["protein", "ISOTOPELABEL_ID", "Sequence"]

    df_quant_results = None  # type: Union[Any, pd.DataFrame]

    for method in methods:
        if df_quant_results is not None:
            df_quant_results_sub = get_maxarea_per_protein(
                df_quant_training, f"top_target_{method}"
            )
            df_quant_results = df_quant_results.merge(
                df_quant_results_sub, on=group_cols, how="outer"
            )
            df_quant_results = df_quant_results.rename(
                {"MAXAREA": f"MAXAREA_{method}"}, axis=1
            )
        else:
            # get all top targets from msgfplus
            df_quant_results = get_maxarea_per_protein(
                df_quant_training, f"top_target_{method}"
            )
            df_quant_results = df_quant_results.rename(
                {"MAXAREA": f"MAXAREA_{method}"}, axis=1
            )

    # fill any NaNs with 0
    df_quant_results = df_quant_results.fillna(0).sort_values(["protein", "Sequence"])
    return df_quant_results


def get_unique_sequence_counts(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
  Calculate the number of unique top-target peptides for each protein returned a given method
  Arguments:
    - df: dataframe containing maxarea values
    - method: name of method to return results for
  Returns:
    - df_seq_counts: dataframe containing number of unique peptides for each protein
                     for a given method
  """
    df_seq_counts = df[df[f"{method}"] > 0].groupby("protein")["Sequence"].nunique()
    df_seq_counts = df_seq_counts.reset_index()
    return df_seq_counts


def get_unique_sequence_counts_all(df: pd.DataFrame) -> pd.DataFrame:
    """
  Calculate the number of unique top-target peptides for each protein returned all methods
  Arguments:
    - df: dataframe containing maxarea values
  Returns:
    - df_seq_counts: dataframe containing number of unique peptides for each protein
  """

    # get columns for each method
    cols = [c for c in df.columns if "MAXAREA_" in c]

    # go over each method and get counts
    df_seq_counts = None  # type: Union[Any, pd.DataFrame]
    for col in cols:
        method = col.split("MAXAREA_")[-1]
        if df_seq_counts is not None:
            df_seq_counts = df_seq_counts.merge(
                get_unique_sequence_counts(df, col), on="protein", how="outer"
            )
            df_seq_counts = df_seq_counts.rename(
                {"Sequence": f"n_sequences_{method}"}, axis=1
            )
        else:
            df_seq_counts = get_unique_sequence_counts(df, col)
            df_seq_counts = df_seq_counts.rename(
                {"Sequence": f"n_sequences_{method}"}, axis=1
            )

    # replace NaNs with 0s
    df_seq_counts = df_seq_counts.fillna(0)
    return df_seq_counts


def calc_channel_ratios(
    df: pd.DataFrame, prot: str, method: str = "msgfplus"
) -> pd.DataFrame:
    """
  Calculate the ratios between each pair of channels for a given protein
  using a given scoring method
  Arguments:
    - df: dataframe containing MAXAREA values for each channel
    - prot: name of protein to calculate the ratios for
    - method: name of method (e.g. search engine name, ML model name)
  Returns:
    - df_prot: dataframe containing ratio values
  """
    # get data for protein 'prot'
    df_prot = df[df["protein"] == prot]
    # only take values greater than 0
    df_prot = df_prot[df_prot[f"MAXAREA_{method}"] > 0]
    # get results for each channel as columns
    df_prot = pd.pivot_table(
        df_prot,
        values=f"MAXAREA_{method}",
        index=["Sequence"],
        columns=["ISOTOPELABEL_ID"],
        aggfunc=np.sum,
    )
    # reset index and make the sequence as the index
    df_prot = df_prot.reset_index()
    df_prot = df_prot.set_index("Sequence")
    del df_prot.index.name
    # go over each pair of columns and get the ratio for each protein
    cols = df_prot.columns
    nc = len(cols)
    for ic1 in np.arange(nc - 1):
        for ic2 in np.arange(ic1 + 1, nc):
            col_name = f"ratio_{cols[ic1]}_{cols[ic2]}"
            df_prot[col_name] = np.log2(df_prot[cols[ic1]] / df_prot[cols[ic2]])
    df_prot = df_prot.reset_index()
    df_prot = df_prot.rename({"index": "Sequence"}, axis=1)
    return df_prot


def calc_mean_channel_ratio_diff(df: pd.DataFrame, prot_list: List) -> pd.DataFrame:
    """
  Calculate ratios in channel values for all methods (search engine and ML)
  Arguments:
    - df: dataframe containing MAXAREA values for each channel
    - prot_list: list of proteins to calculate ratios for
  Returns:
    - df_output: dataframe containing channel ratios for all scoring methods
  """
    # columns to be outputted
    channels = np.arange(62, 72)
    cols = [f"diff_var_SVM_ratio_{i}_{j}" for i in channels for j in channels if j > i]
    cols += [f"diff_var_RF_ratio_{i}_{j}" for i in channels for j in channels if j > i]
    # make an empty dataframe
    df_output = pd.DataFrame(columns=cols, index=prot_list)
    # go through each protein
    for prot in prot_list:
        # calculate the ratios for that protein
        df_prot = calc_channel_ratios(df, prot, method="msgfplus")
        df_prot = df_prot.loc[:, df_prot.notnull().sum() > 3]
        # go through and get the average and standard deviation for each ratio
        cols = [c for c in df_prot.columns if "ratio_" in str(c)]
        test1 = df_prot[cols].agg(["mean", "var"]).T

        # get ratios for SVM, get averages/vars and combine with search engine data
        df_prot = calc_channel_ratios(df, prot, method="SVM")
        df_prot = df_prot.loc[:, df_prot.notnull().sum() > 3]
        cols = [c for c in df_prot.columns if "ratio_" in str(c)]
        test2 = df_prot[cols].agg(["mean", "var"]).T
        test1 = test1.merge(test2, left_index=True, right_index=True, how="outer")
        test1 = test1.rename(
            {
                "mean_x": "mean_msgfplus",
                "var_x": "var_msgfplus",
                "mean_y": "mean_SVM",
                "var_y": "var_SVM",
            },
            axis=1,
        )

        # get ratios for RF, get averages/vars and combine with search engine data
        df_prot = calc_channel_ratios(df, prot, method="RF")
        df_prot = df_prot.loc[:, df_prot.notnull().sum() > 3]
        cols = [c for c in df_prot.columns if "ratio_" in str(c)]
        test2 = df_prot[cols].agg(["mean", "var"]).T
        test1 = test1.merge(test2, left_index=True, right_index=True, how="outer")
        test1 = test1.rename({"mean": "mean_RF", "var": "var_RF"}, axis=1)

        # calculate differences between search engine and RF/SVM
        test1["diff_mean_RF"] = test1["mean_msgfplus"] - test1["mean_RF"]
        test1["diff_mean_SVM"] = test1["mean_msgfplus"] - test1["mean_SVM"]
        test1["diff_var_RF"] = test1["var_msgfplus"] - test1["var_RF"]
        test1["diff_var_SVM"] = test1["var_msgfplus"] - test1["var_SVM"]

        # add data to output dataframe
        df_output.loc[prot, [f"diff_var_SVM_{c}" for c in test1.index]] = test1[
            "diff_var_SVM"
        ].values

        df_output.loc[prot, [f"diff_var_RF_{c}" for c in test1.index]] = test1[
            "diff_var_RF"
        ].values

    return df_output


def calc_all_ratios(
    df: pd.DataFrame, i: int, j: int, prots: List, methods: List
) -> Dict:
    """
  Make a dictionary containing ratios for all psms for all proteins and methods supplied. Data
  is stored in dataframes for each protein, as well as an overall dataframe 'all_data'. Dataframes
  contain ratios and sums of the channel vlaues
  Arguments:
    - df: dataframe containing max_area results
    - i, j: channels to plot, with ratio taken as i/j
    - prots: list of proteins to calculate ratios for
    - methods: list of methods to calculate for (e.g. mascot, SVM)
  Returns:
    - data_ratios: dictionary containing results, sorted into Human and E-Coli proteins
  """

    # column names for ratios and sums
    cols = [f"ratio_{i}_{j}_{m}" for m in methods] + [
        f"sum_{i}_{j}_{m}" for m in methods
    ]

    # set a dictionary to store the data
    data_ratios = {
        "Human": {},
        "E-Coli": {},
    }  # type: Dict[str, Dict[str, pd.DataFrame]]
    # all_data is plotted first, with results for all proteins
    for val in data_ratios.values():
        val["all_data"] = pd.DataFrame(columns=cols)

    # loop over all proteins
    for prot in prots:
        # flag for if data has sufficient values for the given channels
        no_data_flag = False
        ratio = f"ratio_{i}_{j}"
        if "coli" in prot:
            prot_type = "E-Coli"
        else:
            prot_type = "Human"
        vals = None  # type: Union[Any, pd.DataFrame]
        # loop over all methods
        for method in methods:
            # get the ratios for that method
            df_prot = calc_channel_ratios(df, prot, method=method).sort_values(
                "Sequence"
            )
            # if the ratio is not in the columns (i.e.
            # there wasn't enough data, set a flag and break)
            if ratio not in df_prot.columns:
                no_data_flag = True
                break
            # make a dataframe with all the data for each method
            if vals is not None:
                vals = vals.merge(
                    df_prot[["Sequence", i, j, ratio]], on="Sequence", how="outer"
                )
            else:
                vals = df_prot[["Sequence", i, j, ratio]]
            # change the column names
            vals = vals.rename(
                {
                    i: f"{i}_{method}",
                    j: f"{j}_{method}",
                    f"ratio_{i}_{j}": f"ratio_{i}_{j}_{method}",
                },
                axis=1,
            )
            # calculate the sum
            vals[f"sum_{i}_{j}_{method}"] = (
                vals[f"{i}_{method}"] + vals[f"{j}_{method}"]
            )
        if not no_data_flag:
            data_ratios[prot_type][prot] = vals[cols]
            data_ratios[prot_type]["all_data"] = pd.concat(
                [data_ratios[prot_type]["all_data"], data_ratios[prot_type][prot]],
                ignore_index=True,
            )
    return data_ratios


# pylint: disable=too-many-locals
def get_ratio_against_sum(
    df: pd.DataFrame,
    sum_cut: float = 0.0,
    n_slices: int = 15,
    quant_cut: bool = True,
    bin_log_sum: bool = False,
    mean_of_log: bool = False,
    methods: List = None,
    method_0: str = "all-engines",
    q_val_cuts: List = None,
    q_val_cut_fixed: float = None,
    icol: int = 63,
    jcol: int = 69,
) -> pd.DataFrame:
    """
  Find the plateau value for the ratio against sum curve of channels icol and jcol for
  a range of q-values used as the cut-off to identify top targets
  Arguments:
    - df: dataframe containing results from search engines and training + quant data
    - sum_cut: ignore all PSMs with sum of channels i + j <= sum_cut
    - quant_cut: use qcut (quantiles, default) or cut (even sized bins)
    - bin_log_sum: bin the log of the sum of the channels if True,
                   otherwise just bin the sum (default)
    - mean_of_log: calculate the mean of the log of the sum (True),
                   otherwise calculate the log of the mean (default)
    - methods: list methods to calculate data for, default is None (use all)
    - method_0: method to use as 'base' method for  additional PSMs
                (PSMs identified by a method, but not by method_0)
    - q_val_cuts: q-values to use as cut-off when calculating ratios
    - q_val_cut_fixed: use the psms at q_val_cut=q_val_cut_fixed for comparison, default=None (use
                       q_val_cuts)
    - icol, jcol: ratio to calculate, where icol/jcol is taken for the ratio.
  Returns:
    - df_mean_plateau: dataframe containing plateau value of the mean ratio
  """

    # get q-values
    if q_val_cuts is None:
        q_val_cuts = sorted(
            [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(3, 1, -1)]
        ) + [1e-1]

    # get methods
    if methods is None:
        methods = [c.split("top_target_")[-1] for c in df.columns if "top_target" in c]

    columns = sorted(
        [f"log2(<ratio {icol}/{jcol}>) [sum->infty]_{m}" for m in methods + [method_0]]
    )
    columns += [f"error [{c}]" for c in columns]

    df_mean_plateau = pd.DataFrame(columns=columns, index=[str(q) for q in q_val_cuts])

    # loop over q-values to use for the cut-offs
    for q_val_cut in q_val_cuts:

        # get values for method_0
        if q_val_cut_fixed:
            df = get_top_targets(df, q_val_cut=q_val_cut_fixed)
        else:
            df = get_top_targets(df, q_val_cut=q_val_cut)

        # get values for method_0
        mean_ratio_plateau, error, bins = get_mean_error(
            df,
            sum_cut=sum_cut,
            method=method_0,
            icol=icol,
            jcol=jcol,
            quant_cut=quant_cut,
            n_slices=n_slices,
            bin_log_sum=bin_log_sum,
            mean_of_log=mean_of_log,
            bins=None,
        )

        col = f"log2(<ratio {icol}/{jcol}>) [sum->infty]_{method_0}"
        df_mean_plateau.loc[str(q_val_cut), col] = mean_ratio_plateau
        df_mean_plateau.loc[str(q_val_cut), f"error [{col}]"] = error

        # now loop over all methods and do the same, using the bins from all-engines
        df = get_top_targets(df, q_val_cut=q_val_cut)
        for method in methods:
            mean_ratio_plateau, error = get_mean_error(
                df,
                sum_cut=sum_cut,
                method=method,
                icol=icol,
                jcol=jcol,
                quant_cut=quant_cut,
                n_slices=n_slices,
                bin_log_sum=bin_log_sum,
                mean_of_log=mean_of_log,
                bins=bins,
            )[0:2]

            col = f"log2(<ratio {icol}/{jcol}>) [sum->infty]_{method}"
            df_mean_plateau.loc[str(q_val_cut), col] = mean_ratio_plateau
            df_mean_plateau.loc[str(q_val_cut), f"error [{col}]"] = error

        # loop over the differences in the methods
        # (e.g. what is a top-target for RF, but not method_0)
        for method in methods:
            mean_ratio_plateau, error = get_mean_error(
                df,
                sum_cut=sum_cut,
                method=method,
                method_0=method_0,
                icol=icol,
                jcol=jcol,
                quant_cut=quant_cut,
                n_slices=n_slices,
                bin_log_sum=bin_log_sum,
                mean_of_log=mean_of_log,
                bins=bins,
            )[0:2]

            col = f"log2(<ratio {icol}/{jcol}>) [sum->infty]_{method}-from-{method_0}"
            df_mean_plateau.loc[str(q_val_cut), col] = mean_ratio_plateau
            df_mean_plateau.loc[str(q_val_cut), f"error [{col}]"] = error
    return df_mean_plateau


# pylint: disable=too-many-locals
def get_mean_error(
    df: pd.DataFrame,
    method: str,
    icol: int,
    jcol: int,
    bin_log_sum: bool,
    sum_cut: float,
    mean_of_log: bool,
    quant_cut: bool,
    n_slices: int,
    bins: List = None,
    method_0: str = None,
) -> pd.DataFrame:
    """
  Return the plateau value of the ratio between channels icol and jcol, as well
  error as the error, for a given method
  Arguments:
    - df: dataframe containing results from search engines and training + quant data
    - sum_cut: ignore all PSMs with sum of channels i + j <= sum_cut
    - quant_cut: use qcut (quantiles, default) or cut (even sized bins)
    - bin_log_sum: bin the log of the sum of the channels if True,
                   otherwise just bin the sum (default)
    - mean_of_log: calculate the mean of the log of the sum (True),
                   otherwise calculate the log of the mean (default)
    - method: method to calculate data for
    - icol, jcol: ratio to calculate, where icol/jcol is taken for the ratio.
    - method_0: method to compare to, taking PSMs that are identified by method but are not
                by method_0. Default is None (don't use)
  Returns:
    - mean_ratio_plateau: plateau value
    - error: error on the above value
  """

    # get the ratio between channels icol and jcol
    df_ratio = get_ratios_method(
        df, method=None, i=icol, j=jcol, drop_no_data=True, log2=True
    )

    if method_0 is not None:
        df_ratio = df_ratio[
            df_ratio[f"top_target_{method}"] & ~df_ratio[f"top_target_{method_0}"]
        ]

    # keep only top_targets for method
    df_ratio = df_ratio[df_ratio[f"top_target_{method}"]]
    # get the name of the sum column
    sum_col = df_ratio.filter(like="sum").columns[0]
    # get only values above the cut-off
    df_ratio = df_ratio[df_ratio[sum_col] >= sum_cut]
    # get the mean ratio values and the bins (if bins are not sent)
    if bins is not None:
        df_means = get_logn_mean_error(
            df_ratio,
            base=2,
            bins_in=bins,
            bin_log_sum=bin_log_sum,
            mean_of_log=mean_of_log,
        )[0]
    else:
        df_means, bins = get_logn_mean_error(
            df_ratio,
            base=2,
            quant_cut=quant_cut,
            n_slices=n_slices,
            bin_log_sum=bin_log_sum,
            mean_of_log=mean_of_log,
        )

    mean_ratio_plateau, error = fit_function(df_means)[:2]
    return mean_ratio_plateau, error, bins
