"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Plot results from iterative fitting models
"""

# pylint: disable=unused-import
import contextlib
from textwrap import wrap
from typing import Any, Dict, Pattern, Set, Tuple, List, Generator, Union

import sys

# if sys.platform.startswith('darwin'):
#  # required for mac osx
#  import matplotlib
#  matplotlib.use('TkAgg')

from matplotlib import colors as mcolors

# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from .analyse_results import (
    get_protein_maxarea_dataset,
    get_num_psms_by_method,
    get_num_psms_against_q_cut,
    get_top_targets,
    get_ranks,
)
from .quant_data_verification import (
    get_unique_sequence_counts_all,
    calc_all_ratios,
    get_ratios_method,
    get_logn_mean_error,
    atan_func,
    fit_function,
    get_ratio_against_sum,
)

# from utils.percolator_utils import get_perc_results


@contextlib.contextmanager
def dummy_context_mgr() -> Generator[None, None, None]:
    """
  Dummy to use with 'with' statement
  """
    yield None


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def plot_psm_numbers(
    psms: Dict,
    psms_avg: Dict,
    psms_engine: Dict,
    psms_other: Dict = None,
    plot_train: bool = True,
    plot_val: bool = False,
    perc_file: str = None,
    perc_val_file: str = None,
    align_horizontal: bool = False,
    output_file: str = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> None:
    """
  Plot the number of PSMs against number of iterations
  Arguments:
    - psms: dictionary containing psms of the format {model_name: results},
            where results has the format {split_name: [list of no. of PSMs identified]}
    - psms_avg: same as above, but with PSMs identified by the score averaged over multiple
                iterations
    - psms_engine: dictionary containing PSMs identified by the search engine. Has
                   the format {split_name: no. of PSMs identified}
    - psms_other: dictionary other results that are to be plotted. Format is '{method}': {num_psms}
    - plot_train: plot results for the training set
    - plot_val: plot the validation results, default = False
    - perc_file: filename for results from percolator. Default = None
    - perc_val_file: filename for results from percolator on validation set.
                     Default = None
    - align_horizontal: place plots side by side (if validation data is plotted). Default = False,
                        which places the plots on top of each other
    - output_file: file to save plot to. If = None (default), do not save the plot
    - dpi: plot output quality, dots per inch. Default is 300
  """
    # get dictionary of colors
    colors_list = list(mcolors.BASE_COLORS.keys())
    # make list of models to plot data for
    models = list(psms.keys())
    engines = list(psms_engine.keys())
    # get one color per model
    colors = {model: color for model, color in zip(models + engines, colors_list)}

    max_len_x = 0

    # initialize the plot
    if plot_val:
        if align_horizontal:
            plt.figure(num=None, figsize=(24, 7), dpi=dpi, facecolor="w", edgecolor="k")
            plt.subplot(1, 2, 1)
        else:
            plt.figure(
                num=None, figsize=(15, 15), dpi=dpi, facecolor="w", edgecolor="k"
            )
            plt.subplot(2, 1, 1)
    else:
        plt.figure(num=None, figsize=(13, 9), dpi=dpi, facecolor="w", edgecolor="k")

    for model in models:
        model_sub = model.split(" -")[0]
        test_label = model_sub
        if plot_train:
            # get data for x-axis: no. of iterations
            x = np.arange(1, len(psms[model]["train"]) + 1)
            max_len_x = max(max_len_x, max(x))
            n_train = len(psms[model]["train"]) - len(psms_avg[model]["train"])
            x_avg = np.arange(n_train + 1, n_train + len(psms_avg[model]["train"]) + 1)
            plt.plot(
                x,
                psms[model]["train"],
                linewidth=5,
                linestyle="--",
                color=colors[model],
                label=f"{model_sub} - train",
            )
            plt.plot(x_avg, psms_avg[model]["train"], linewidth=5, color=colors[model])
            test_label += " - test"
        x = np.arange(1, len(psms[model]["test"]) + 1)
        max_len_x = max(max_len_x, max(x))
        n_train = len(psms[model]["test"]) - len(psms_avg[model]["test"])
        x_avg = np.arange(n_train + 1, n_train + len(psms_avg[model]["test"]) + 1)
        plt.plot(
            x,
            psms[model]["test"],
            linestyle=":",
            linewidth=5,
            color=colors[model],
            label=test_label,
        )
        plt.plot(x_avg, psms_avg[model]["test"], linewidth=5, color=colors[model])

    x_other = [1, x[-1]]
    if psms_other:
        for k, v in psms_other.items():
            plt.plot(x_other, [v, v], linestyle="-.", linewidth=5, label=k)

    for engine, data in psms_engine.items():
        label = engine
        for k, v in data.items():
            if plot_train:
                label = f"{label} - {k}"
            if k != "train":
                plt.plot(
                    x_other,
                    [v for i in x_other],
                    linewidth=5,
                    linestyle="-.",
                    color=colors[engine],
                    label=label,
                )
            else:
                if plot_train:
                    plt.plot(
                        x_other,
                        [psms_engine["train"] for i in x_other],
                        linewidth=5,
                        linestyle="--",
                        color=colors[engine],
                        label=label,
                    )

    if perc_file:
        perc_train, perc_test, perc_test_non_redundent = get_perc_results(perc_file)
        test_label = "percolator"
        if plot_train:
            y = [p for p in perc_train] + [
                perc_train[-1] for i in np.arange(max_len_x - len(perc_train))
            ]
            plt.plot(
                np.arange(1, len(y) + 1),
                y,
                linewidth=5,
                linestyle="--",
                color="black",
                label="percolator - train",
            )
            test_label = f"{test_label} - test"

        y = [p for p in perc_test] + [
            perc_test[-1] for i in np.arange(max_len_x - len(perc_test))
        ]
        plt.plot(
            np.arange(1, len(y) + 1),
            y,
            linewidth=5,
            color="black",
            linestyle=":",
            label=test_label,
        )

        n_points = len(x) - len(perc_test_non_redundent)
        y = [p for p in perc_test_non_redundent] + [
            perc_test_non_redundent[-1] for i in np.arange(n_points)
        ]
        plt.plot(
            np.arange(1, len(y) + 1),
            y,
            linewidth=5,
            color="black",
            linestyle="-",
            label=f"{test_label} (non-redundent)",
        )

    plt.legend(
        bbox_to_anchor=(0.0, -0.15), loc=2, borderaxespad=0.0, fontsize=16, ncol=3
    )
    plt.xlabel("iterations", fontsize=18)
    plt.ylabel("PSMs below q=1%", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Number of Target PSMs identified at each iteration", fontsize=18)

    if plot_val:
        if align_horizontal:
            plt.subplot(1, 2, 2)
            plt.title(
                "Number of Target PSMs identified at each iteration: unseen data",
                fontsize=18,
            )
        else:
            plt.subplot(2, 1, 2)
        plt.plot(
            x,
            [psms_engine["val"] for i in x],
            linewidth=5,
            color="orange",
            linestyle="--",
            label="engine - val",
        )
        if perc_val_file:
            _, perc_val_test, _ = get_perc_results(perc_val_file)
            plt.plot(
                x,
                [p for p in perc_val_test]
                + [perc_val_test[-1] for i in np.arange(len(x) - len(perc_val_test))],
                linewidth=5,
                color="black",
                linestyle="--",
                label="percolator - val",
            )
        for model in models:
            model_sub = model.split(" -")[0]
            plt.plot(
                x,
                psms[model]["val"],
                linestyle="--",
                linewidth=5,
                color=colors[model],
                label=f"{model_sub} - val",
            )
            plt.plot(x_avg, psms_avg[model]["val"], linewidth=5, color=colors[model])

        plt.legend(bbox_to_anchor=(1.005, 0.95), loc=2, borderaxespad=0.0, fontsize=18)
        plt.xlabel("iterations", fontsize=18)
        plt.ylabel("PSMs below q=1%", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    plt.tight_layout()
    if output_file:
        file_format = output_file.split(".")[-1]
        plt.savefig(output_file, format=file_format, dpi=dpi)
    if show_plot:
        plt.show()


def plot_accuracies(
    scores: Dict[str, Dict[str, Dict]], scores_1pc: Dict[str, Dict[str, Dict]] = None
) -> None:
    """
  Plot the accuracies for targets, decoys and in total. If provided, also
  plot the accuracies for the top 1% of targets.
  Arguments:
    - scores: dictionary containing scores for training, test and validation
              sets, for targets, decoys and all PSMs, and for each model. Has
              format scores[model_type][split][target/decoy/all]
    - scores_1pc: same as above, but only using top 1% of targets
  """

    # get dictionary of colors
    colors_list = list(mcolors.BASE_COLORS.keys())
    # make list of models to plot data for
    models = list(scores.keys())
    # and a list of splits (train/test/val)
    splits = scores[models[0]].keys()
    # get one color per model
    colors = {model: color for model, color in zip(models, colors_list)}

    for split in splits:
        # start the plot
        plt.figure(num=None, figsize=(13, 8), dpi=300, facecolor="w", edgecolor="k")
        for model in models:
            model_sub = model.split(" -")[0]
            yt = scores[model][split]["targets"]
            yd = scores[model][split]["decoys"]
            y = scores[model][split]["all"]
            x = [(i + 1) for i in np.arange(len(y))]
            plt.plot(x, y, linewidth=5, color=colors[model], label=f"{model_sub} - all")
            plt.plot(
                x,
                yt,
                linestyle="--",
                linewidth=5,
                color=colors[model],
                label=f"{model_sub} - targets",
            )
            plt.plot(
                x,
                yd,
                linestyle=":",
                linewidth=5,
                color=colors[model],
                label=f"{model_sub} - decoys",
            )

        plt.legend(bbox_to_anchor=(1.005, 0.95), loc=2, borderaxespad=0.0, fontsize=18)
        plt.xlabel("iterations", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f"Accuracy of model for each iteration: {split} split", fontsize=18)
    plt.show()

    if scores_1pc:
        for split in splits:
            # start the plot
            plt.figure(num=None, figsize=(13, 8), dpi=300, facecolor="w", edgecolor="k")
            for model in models:
                model_sub = model.split(" -")[0]
                yt = scores_1pc[model][split]["targets"]
                yd = scores_1pc[model][split]["decoys"]
                y = scores_1pc[model][split]["all"]
                x = [(i + 1) for i in np.arange(len(y))]
                plt.plot(
                    x, y, linewidth=5, color=colors[model], label=f"{model_sub} - all"
                )
                plt.plot(
                    x,
                    yt,
                    linestyle="--",
                    linewidth=5,
                    color=colors[model],
                    label=f"{model_sub} - targets",
                )
                plt.plot(
                    x,
                    yd,
                    linestyle=":",
                    linewidth=5,
                    color=colors[model],
                    label=f"{model_sub} - decoys",
                )

            plt.legend(
                bbox_to_anchor=(1.005, 0.95), loc=2, borderaxespad=0.0, fontsize=18
            )
            plt.xlabel("iterations", fontsize=18)
            plt.ylabel("Accuracy", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(
                f"Accuracy of model for each iteration, q < 1%: {split} split",
                fontsize=18,
            )
        plt.show()


def plot_ranks(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    use_top_psm: bool = True,
    n_psms: int = 6,
    output_file: str = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> None:
    """
  Plot ranks of PSMs from two methods against each other
  Arguments:
    - df: dataframe containing ranks from different methods
    - x_name: name of method for the x-axis
    - y_name: name of method for the y-axis
    - use_top_psm: use the top PSMs for each spectrum (True, default), otherwise use all PSMs
    - n_psms: number of psms to plot results for, in multiples of the number with q-value<1%
    - output_file: name of output file for image. If None (default), don't save the image
    - dpi: dots per inch for output file (defaul = 300)
  """
    # initiate the plot
    plt.figure(num=None, figsize=(13, 13), dpi=dpi, facecolor="w", edgecolor="k")

    # names of columns to use
    col_x = f"rank_{x_name}"
    tt_x = f"top_target_{x_name}"

    col_y = f"rank_{y_name}"
    tt_y = f"top_target_{y_name}"

    if use_top_psm:
        df_plot = df.sort_values(
            f"Score_processed_{y_name}", ascending=False
        ).drop_duplicates("Spectrum ID")
        df_plot = get_ranks(df_plot, from_scores=True)
    else:
        df_plot = df.copy(deep=True)

    # cut off values for q<1%
    x_cut = sum(df_plot[tt_x])
    x_cut = [x_cut for i in range(2)]
    y_cut = sum(df_plot[tt_y])
    y_cut = [y_cut for i in range(2)]

    # set the max limits to plot
    x_max = min(round(n_psms * x_cut[0], 1), round(len(df_plot), 1))
    y_max = x_max

    # set the diagonal line y = x
    x_diag = np.linspace(0, x_max, 2)
    y_diag = np.linspace(0, y_max, 2)

    # plot the diagonal lines and the cut offs
    plt.plot(x_diag, y_diag, color="grey", lw=4, zorder=6, alpha=0.7)
    plt.plot(x_diag, y_cut, color="black", lw=4, zorder=5, alpha=0.5)
    plt.plot(x_cut, y_diag, color="black", lw=4, zorder=4, alpha=0.5)

    # plot top targets from y_name method
    x = df_plot.loc[df_plot[tt_y], col_x]
    y = df_plot.loc[df_plot[tt_y], col_y]
    plt.plot(
        x,
        y,
        linestyle="None",
        marker=".",
        ms=4,
        color="orange",
        alpha=0.75,
        zorder=1,
        label="Top Targets",
    )

    # plot other targets
    x = df_plot.loc[~df_plot[tt_y] & ~df_plot["Is decoy"], col_x]
    y = df_plot.loc[~df_plot[tt_y] & ~df_plot["Is decoy"], col_y]
    plt.plot(
        x,
        y,
        linestyle="None",
        marker=".",
        ms=4,
        color="red",
        alpha=0.75,
        zorder=2,
        label="Targets",
    )

    # plot other targets that were top targets before
    x = df_plot.loc[~df_plot[tt_y] & df_plot[tt_x], col_x]
    y = df_plot.loc[~df_plot[tt_y] & df_plot[tt_x], col_y]
    plt.plot(
        x,
        y,
        linestyle="None",
        marker=".",
        ms=4,
        color="cyan",
        alpha=0.75,
        zorder=3,
        label="Former Top Targets",
    )

    # plot decoys
    x = df_plot.loc[~df_plot[tt_y] & df_plot["Is decoy"], col_x]
    y = df_plot.loc[~df_plot[tt_y] & df_plot["Is decoy"], col_y]
    plt.plot(
        x,
        y,
        linestyle="None",
        marker=".",
        ms=4,
        color="blue",
        alpha=0.75,
        zorder=3,
        label="Decoys",
    )

    # make a legend
    lgnd = plt.legend(
        bbox_to_anchor=(1.005, 0.95), loc=2, borderaxespad=0.0, fontsize=14
    )
    # pylint: disable=W0212
    lgnd.legendHandles[0]._legmarker.set_markersize(12)
    # pylint: disable=W0212
    lgnd.legendHandles[1]._legmarker.set_markersize(12)
    # pylint: disable=W0212
    lgnd.legendHandles[2]._legmarker.set_markersize(12)

    # set axis, limits, tics etc.
    plt.axis("equal")
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(col_x, fontsize=18)
    plt.ylabel(col_y, fontsize=18)
    plt.title(f"Rank from {y_name} vs rank from {x_name}", fontsize=18)

    # save the image if a name is given
    if output_file:
        file_format = output_file.split(".")[-1]
        plt.savefig(output_file, format=file_format, dpi=dpi)
    # display the image
    if show_plot:
        plt.show()


def plot_maxarea(df: pd.DataFrame, prot: str) -> None:
    """
  Plot the max area values for a given protein, for each method
  Arguments:
    - df: dataframe with the max area for each peptide for each protein
    - prot: name of the protein ID to use
  """
    df_maxarea = get_protein_maxarea_dataset(df, prot)
    g = sns.catplot(
        x="ISOTOPELABEL_ID",
        y="MAXAREA",
        hue="Sequence",
        col="method",
        data=df_maxarea,
        kind="bar",
        height=8,
        aspect=1.1,
        col_wrap=2,
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Protein: {prot[0:50]}", fontsize=18)
    plt.show()


def plot_channel_sum_against_ratio(
    df: pd.DataFrame,
    i: int,
    j: int,
    prots: List = None,
    output_file: str = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
  Plot the ratio of a channel i against channel j for each peptide for each protein
  Arguments:
    - df: dataframe containing max_area results
    - i, j: channels to plot, with ratio taken as i/j
    - prots: list of proteins to plot. If None (default), plot all proteins with
             more than two peptides from at least one method
    - output_file: name of file to save plots to, as a single pdf. Default is None
                   (don't plot). If is None, then show_plots is set to True
    - show_plots: show the plots (default = False)
  """
    methods = [c.split("MAXAREA_")[-1] for c in df if "MAXAREA_" in c]
    cols = [f"ratio_{i}_{j}_{m}" for m in methods]
    cols_sum = [f"sum_{i}_{j}_{m}" for m in methods]

    if output_file:
        output_file = output_file.split(".pdf")[0]
        output_file = f"{output_file}_ratio_{i}_{j}"
    else:
        show_plots = True

    if prots is None:
        df_seq_counts = get_unique_sequence_counts_all(df)
        seq_count_cols = [c for c in df_seq_counts if "n_sequences" in c]
        df_seq_counts = df_seq_counts[(df_seq_counts[seq_count_cols] > 1.0).any(axis=1)]
        prots = [p for p in df_seq_counts["protein"].unique()]

    # sort the proteins into human and e-coli, in order to order them
    prots_ecoli = [p for p in prots if "coli" in p.lower()]
    prots_human = [p for p in prots if "human" in p.lower()]
    prots = prots_human + prots_ecoli

    print(f"Plotting data for {len(prots)} proteins:")
    print(f"{len(prots_human)} Human and {len(prots_ecoli)} E-Coli")

    data_plot = calc_all_ratios(df, i, j, prots, methods)

    x_min = min(
        data_plot["Human"]["all_data"][cols].min().min(),
        data_plot["E-Coli"]["all_data"][cols].min().min(),
    )
    x_max = max(
        data_plot["Human"]["all_data"][cols].max().max(),
        data_plot["E-Coli"]["all_data"][cols].max().max(),
    )
    y_min = min(
        data_plot["Human"]["all_data"][cols_sum].min().min(),
        data_plot["E-Coli"]["all_data"][cols_sum].min().min(),
    )
    y_max = max(
        data_plot["Human"]["all_data"][cols_sum].max().max(),
        data_plot["E-Coli"]["all_data"][cols_sum].max().max(),
    )

    for k, v in data_plot.items():

        with PdfPages(
            f"{output_file}_{k}.pdf"
        ) if output_file else dummy_context_mgr() as pp:
            for k1, v1 in v.items():
                plt_tmp = plt.figure(
                    num=None, figsize=(10, 7), dpi=300, facecolor="w", edgecolor="k"
                )
                plt.title(
                    "\n".join(wrap(k1, 80)).replace("<|>", "\n\n") + "\n", fontsize=18
                )
                colors = {
                    "mascot": "purple",
                    "msgfplus": "orange",
                    "SVM": "green",
                    "RF": "black",
                }
                markers = {"mascot": "v", "msgfplus": "s", "SVM": "o", "RF": "x"}
                mss = {"mascot": 16, "msgfplus": 14, "SVM": 10, "RF": 10}
                mews = {"mascot": 1, "msgfplus": 1, "SVM": 1, "RF": 1.5}
                for m in ["mascot", "msgfplus", "SVM", "RF"]:
                    plt.plot(
                        v1[f"ratio_{i}_{j}_{m}"].values,
                        v1[f"sum_{i}_{j}_{m}"].values,
                        color=colors[m],
                        ms=mss[m],
                        mew=mews[m],
                        marker=markers[m],
                        linestyle="None",
                        label=f"{m}",
                    )
                plt.legend(
                    bbox_to_anchor=(1.005, 0.95), loc=2, borderaxespad=0.0, fontsize=14
                )
                plt.ylim(max(y_min / 1.01, 0.0), y_max * 1.01)
                plt.xlim(x_min / 1.01, x_max * 1.01)
                plt.ylabel(f"sum {i}+{j}", fontsize=14)
                plt.xlabel(f"ratio {i}/{j}", fontsize=14)
                if output_file:
                    pp.savefig(plt_tmp, bbox_inches="tight")
                if show_plots:
                    plt.show()
                else:
                    plt.close(plt_tmp)


def plot_num_psms_by_method(
    df: pd.DataFrame,
    q_val_cut: float = 0.01,
    methods: List = None,
    output_file: str = None,
    print_values: bool = False,
    dpi: int = 300,
    show_plot: bool = True,
) -> None:
    """
  Plot the number of PSMs for each method that we have results for
  Arguments:
    - df: dataframe containing results from search engines and ML training
    - q_val_cut: q-value used to identify top-targets
    - methods: list of methods to use, if None, use all methods
    - output_file: name of file to save plot to. If None, don't save the image
    - print_values: print the values above the plot, default = False (don't print)
    - dpi: dots per inch to use for the output plot, default = 300
  """
    # get the number of PSMs for each method
    df_num_psms = get_num_psms_by_method(df, methods=methods, q_val_cut=q_val_cut)

    # initiate the plot
    plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
    ax = sns.barplot(x="method", y="n_psms", data=df_num_psms)
    if print_values:
        for i, n_psms in enumerate(df_num_psms["n_psms"].values):
            ax.text(i, n_psms + 200, n_psms, ha="center", fontsize=16)

    ax.set_xlabel("Method", fontsize=16)
    ax.set_ylabel("Num. PSMs", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=2)
    ax.set_title("Number of target PSMs with q-val < 1% by method", fontsize=18)
    plt.tight_layout()

    if output_file:
        file_format = output_file.split(".")[-1]
        plt.savefig(output_file, format=file_format, dpi=dpi)
    if show_plot:
        plt.show()


def plot_num_psms_against_q(
    df: pd.DataFrame,
    q_val_cuts: List = None,
    methods: List = None,
    from_method: str = None,
    output_file: str = None,
    dpi: int = 300,
    show_plot: bool = False,
) -> None:
    """
  Plot the number of PSMs for each method that we have results for
  Arguments:
    - df: dataframe containing results from search engines and ML training
    - q_val_cuts: list of q-values used to identify top-targets
    - methods: list of methods to use, if None, use all methods
    - from_method: method which was used to originally rank the PSMs, to be used here
                   as the second ranking column (default is None, don't use)
    - output_file: name of file to save plot to. If None, don't save the image
    - dpi: dots per inch to use for the output plot, default = 300
  """

    # get the q-value list
    if q_val_cuts is None:
        q_val_cuts = sorted(
            [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(4, 1, -1)]
        ) + [1e-1]

    # get dataframe for number of PSMs as a function of q-value
    df_num_psms_q = get_num_psms_against_q_cut(
        df, methods=methods, from_method=from_method, q_val_cuts=q_val_cuts
    )

    # markers to be used for plotting
    markers_base = [
        "s",
        "o",
        "v",
        "x",
        "^",
        "P",
        "d",
        "h",
        "*",
        "1",
        "2",
        "3",
        "4",
        "<",
        ">",
        "H",
        "p",
        "P",
    ]

    # initiate the plot
    plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
    plt.title(f"Number of PSMs identified against q-value used as cut-off", fontsize=18)
    x = q_val_cuts
    methods = [m.split("top_target_")[-1] for m in df_num_psms_q]
    for i, method in enumerate(methods):
        y = df_num_psms_q[f"top_target_{method}"]
        plt.plot(x, y, linewidth=4, ms=14, marker=markers_base[i], label=method)

    plt.legend(
        bbox_to_anchor=(0.0, -0.15), loc=2, borderaxespad=0.0, fontsize=18, ncol=5
    )
    plt.xlabel(f"q-val cut-off", fontsize=16)
    plt.ylabel(f"Nr. PSMs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale("log")
    plt.tight_layout()
    if output_file:
        file_format = output_file.split(".")[-1]
        plt.savefig(output_file, format=file_format, dpi=dpi)
    if show_plot:
        plt.show()

    # re-calculate the top-targets using q_cut = 1%
    cols = [c for c in df if "top_target_" in c]
    df = df.drop(cols, axis=1)
    df = get_top_targets(df, q_val_cut=0.01)
    # ^---- why ?


# pylint: disable=too-many-arguments
def plot_ratio_against_sum(
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
    ratios: List[List] = None,
    plot_fits: bool = True,
    output_file: str = None,
    dpi: int = 300,
) -> None:
    """
  Plot the ratio between two channels against the sum of those channels
  Arguments:
    - df: dataframe containing results from search engines and training + quant data
    - sum_cut: ignore all PSMs with sum of channels i + j <= sum_cut
    - quant_cut: use qcut (quantiles, default) or cut (even sized bins)
    - bin_log_sum: bin the log of the sum of the channels if True,
                   otherwise just bin the sum (default)
    - mean_of_log: calculate the mean of the log of the sum (True),
                   otherwise calculate the log of the mean (default)
    - methods: list methods to plot data for, default is None (plot all)
    - method_0: method to use as 'base' method when plotting additional PSMs
                (PSMs identified by a method, but not by method_0)
    - q_val_cuts: q-values to use as cut-off when plotting. One plot is made
                  for each q-value
    - q_val_cut_fixed: use the psms at q_val_cut=q_val_cut_fixed for comparison, default=None (use
                       q_val_cuts)
    - ratios: ratios to plot, in form [[i1, j1], [i2, j2], ...]. A plot is made
              for each of pair, using i/j for the ratio.
    - plot_fits: also plot a best-fit curve to the data, default=True
    - output_file: name of file to save plot to. If None, don't save the image
    - dpi: dots per inch to use for the output plot, default = 300
  """
    markers_base = [
        "s",
        "o",
        "v",
        "x",
        "^",
        "P",
        "d",
        "h",
        "*",
        "1",
        "2",
        "3",
        "4",
        "<",
        ">",
        "H",
        "p",
        "P",
    ]

    if ratios is None:
        ratios = [[63, 69]]

    if q_val_cuts is None:
        q_val_cuts = [1e-3, 1e-2]

    if methods is None:
        methods = [c.split("top_target_")[-1] for c in df if "top_target" in c]

    # get a list of colors to plot with
    colormap = plt.cm.jet
    colors = [colormap(i) for i in np.linspace(0, 1, 2 * len(methods) + 1)]

    if q_val_cut_fixed:
        df_fixed = df.copy(deep=True)
        cols = [c for c in df_fixed if "top_target" in c]
        df_fixed = df_fixed.drop(cols, axis=1)
        df_fixed = get_top_targets(df_fixed, q_val_cut=q_val_cut_fixed)

    # loop over q-values to use for the cut-offs
    for q_val_cut in q_val_cuts:
        # get top-targets for a given q-value cut-off
        cols = [c for c in df if "top_target" in c]
        df = df.drop(cols, axis=1)
        df = get_top_targets(df, q_val_cut=q_val_cut)

        # loop over pairs of channels ([i, j] gives ratio i/j)
        for icol, jcol in ratios:
            # base to use for the log, use 2
            base = 2
            # initiate the plot
            plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
            plt.title(
                f"Ratio of channels {icol} and {jcol} against sum of channels: "
                f"E-Coli, q_val_cut={q_val_cut}",
                fontsize=18,
            )

            # get data for method_0, use bins to bin the other methods
            # get the ratio between channels icol and jcol
            if q_val_cut_fixed:
                df_ratio_0 = get_ratios_method(
                    df_fixed, method=None, i=icol, j=jcol, drop_no_data=True, log2=True
                )
                df_ratio = df_ratio_0.copy(deep=True)
            else:
                df_ratio = get_ratios_method(
                    df, method=None, i=icol, j=jcol, drop_no_data=True, log2=True
                )

            # keep only top_targets for all-engines
            df_ratio = df_ratio[df_ratio[f"top_target_{method_0}"]]
            # get the name of the sum column
            sum_col = df_ratio.filter(like="sum").columns[0]
            # get only values above the cut-off
            df_ratio = df_ratio[df_ratio[sum_col] >= sum_cut]
            # get the mean ratio values and the bins
            df_means, bins = get_logn_mean_error(
                df_ratio,
                base=base,
                quant_cut=quant_cut,
                n_slices=n_slices,
                bin_log_sum=bin_log_sum,
                mean_of_log=mean_of_log,
            )

            y_col = df_means.filter(like="ratio").columns[0]
            yerr_col = df_means.filter(like="error").columns[0]
            x_col = df_means.filter(like="sum").columns[0]
            if q_val_cut_fixed:
                label = (
                    f"{method_0}: all PSMs at {q_val_cut_fixed}: {len(df_ratio)} PSMs"
                )
            else:
                label = f"{method_0}: all PSMs at {q_val_cut}: {len(df_ratio)} PSMs"
            plt.errorbar(
                (1e7 / df_means[x_col]),
                df_means[y_col],
                df_means[yerr_col],
                ms=14,
                marker=markers_base[0],
                label=label,
                color="black",
                linestyle="None",
            )

            if plot_fits:
                mean_ratio_0, error, popt = fit_function(df_means)
                plt.plot(
                    (1e7 / df_means[x_col]),
                    atan_func(df_means[x_col], popt[0], popt[1], popt[2]),
                    linewidth=4,
                    color="black",
                    label="_nolegend_",
                )
                print(
                    f"{method_0}: asymptote = {round(mean_ratio_0, 3)} +/ {round(error, 3)}"
                )

            # now loop over all methods and do the same, using the bins from all-engines
            for i, method in enumerate(methods):
                df_ratio = get_ratios_method(
                    df, method=method, i=icol, j=jcol, drop_no_data=True, log2=True
                )
                sum_col = df_ratio.filter(like="sum").columns[0]
                df_ratio = df_ratio[df_ratio[sum_col] >= sum_cut]
                df_means = get_logn_mean_error(
                    df_ratio,
                    base=base,
                    quant_cut=quant_cut,
                    n_slices=n_slices,
                    bins_in=bins,
                    bin_log_sum=bin_log_sum,
                    mean_of_log=mean_of_log,
                )[0]
                y_col = df_means.filter(like="ratio").columns[0]
                yerr_col = df_means.filter(like="error").columns[0]
                x_col = df_means.filter(like="sum").columns[0]

                plt.errorbar(
                    (1e7 / df_means[x_col]),
                    df_means[y_col],
                    df_means[yerr_col],
                    ms=14,
                    marker=markers_base[i + 1],
                    label=f"{method}: all PSMs: {len(df_ratio)} PSMs",
                    color=colors[i + 1],
                    linestyle="None",
                )

                if plot_fits:
                    mean_ratio_0, error, popt = fit_function(df_means)
                    plt.plot(
                        (1e7 / df_means[x_col]),
                        atan_func(df_means[x_col], popt[0], popt[1], popt[2]),
                        linewidth=4,
                        color=colors[i + 1],
                        label="_nolegend_",
                    )
                    print(
                        f"{method}: asymptote = {round(mean_ratio_0, 3)} +/ {round(error, 3)}"
                    )

                # loop over the differences in the methods
                # (e.g. what is a top-target for RF, but not method_0)
                # for i_sub, method in enumerate(methods):
                i2 = i + len(methods)
                df_ratio = get_ratios_method(
                    df, method=None, i=icol, j=jcol, drop_no_data=True, log2=True
                )
                if q_val_cut_fixed:
                    if f"top_target_{method_0}" in df_ratio.columns:
                        df_ratio = df_ratio.drop(f"top_target_{method_0}", axis=1)
                    # print(len(df_ratio))
                    df_ratio = df_ratio.merge(
                        df_ratio_0[[f"top_target_{method_0}"]],
                        left_index=True,
                        right_index=True,
                        how="inner",
                    )
                    df_ratio = df_ratio.fillna(False)
                    # print(len(df_ratio))
                    # .loc[:, f'top_target_{method_0}'] = df_ratio_0.loc[:, f'top_target_{method_0}']
                # print(method, len(df_ratio), len(df_ratio_0))
                # print(df_ratio.shape)
                # print(df_ratio[f'top_target_{method}'].sum())
                # print(df_ratio[f'top_target_{method_0}'].sum())
                # print(df_ratio[f'top_target_{method_0}'].head())
                # print((~df_ratio[f'top_target_{method_0}']).astype(bool).sum())

                df_ratio = df_ratio[
                    df_ratio[f"top_target_{method}"]
                    & ~df_ratio[f"top_target_{method_0}"]
                ]
                sum_col = df_ratio.filter(like="sum").columns[0]
                df_ratio = df_ratio[df_ratio[sum_col] >= sum_cut]
                df_means = get_logn_mean_error(
                    df_ratio,
                    base=base,
                    quant_cut=quant_cut,
                    n_slices=n_slices,
                    bins_in=bins,
                    bin_log_sum=bin_log_sum,
                    mean_of_log=mean_of_log,
                )[0]
                y_col = df_means.filter(like="ratio").columns[0]
                yerr_col = df_means.filter(like="error").columns[0]
                x_col = df_means.filter(like="sum").columns[0]

                plt.errorbar(
                    (1e7 / df_means[x_col]),
                    df_means[y_col],
                    df_means[yerr_col],
                    ms=14,
                    marker=markers_base[i2 + 1],
                    label=f"{method}: only additional PSMs: {len(df_ratio)} PSMs",
                    color=colors[i2 + 1],
                    linestyle="None",
                )

                if plot_fits:
                    mean_ratio_0, error, popt = fit_function(df_means)
                    plt.plot(
                        (1e7 / df_means[x_col]),
                        atan_func(df_means[x_col], popt[0], popt[1], popt[2]),
                        linewidth=4,
                        color=colors[i2 + 1],
                        label="_nolegend_",
                    )
                    print(
                        f"{method}: only additional PSMs: asymptote = {round(mean_ratio_0, 3)}"
                        f" +/ {round(error, 3)}"
                    )

            handles, labels = plt.gca().get_legend_handles_labels()
            nin = len(methods)
            order = (
                np.arange(1, 2 * nin, 2).tolist()
                + [0]
                + np.arange(2, 2 * nin + 1, 2).tolist()
            )
            plt.legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                bbox_to_anchor=(0.0, -0.15),
                loc=2,
                borderaxespad=0.0,
                fontsize=18,
                ncol=2,
            )
            plt.xscale("log")
            plt.xlabel(f"1e7 / {x_col}", fontsize=16)
            plt.ylabel(f"{y_col}", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            if output_file:
                file_base, file_format = output_file.split(".")
                q_val_cut_print_org = str(q_val_cut).split(".")
                q_val_cut_print = f"{q_val_cut_print_org[0]}d{q_val_cut_print_org[1]}"
                output_file_out = (
                    f"{file_base}-{icol}_{jcol}-{q_val_cut_print}.{file_format}"
                )
                plt.savefig(output_file_out, format=file_format, dpi=dpi)
            plt.show()


# pylint: disable=too-many-arguments
def plot_plateau_against_q(
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
    output_file: str = None,
    dpi: int = 300,
) -> None:
    """
  Plot the plateau value of the ratio between two channels against the sum of
  those channels.
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
    - output_file: name of file to save plot to. If None, don't save the image
    - dpi: dots per inch to use for the output plot, default = 300
  """

    # get the data to plot
    df_mean_plateau = get_ratio_against_sum(
        df,
        sum_cut=sum_cut,
        n_slices=n_slices,
        quant_cut=quant_cut,
        bin_log_sum=bin_log_sum,
        mean_of_log=mean_of_log,
        methods=methods,
        method_0=method_0,
        q_val_cuts=q_val_cuts,
        q_val_cut_fixed=q_val_cut_fixed,
        icol=icol,
        jcol=jcol,
    )
    # list of markers to use when plotting
    markers_base = [
        "s",
        "o",
        "v",
        "x",
        "^",
        "P",
        "d",
        "h",
        "*",
        "1",
        "2",
        "3",
        "4",
        "<",
        ">",
        "H",
        "p",
        "P",
    ]
    # get a list of the methods
    methods = [c.split("_")[-1].split("]")[0] for c in df_mean_plateau if "error" in c]
    # base name of each column
    plot_col = f"log2(<ratio {icol}/{jcol}>) [sum->infty]"
    # get the x-values
    x = [float(i) for i in df_mean_plateau.index]
    # initialize the plot
    plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
    for i, method in enumerate(methods):
        col = f"{plot_col}_{method}"
        error_col = f"error [{col}]"
        y = df_mean_plateau[col].values
        yerr = df_mean_plateau[error_col].values
        plt.errorbar(x, y, yerr, ms=14, marker=markers_base[i], label=f"{method}")
    plt.legend(
        bbox_to_anchor=(0.0, -0.15), loc=2, borderaxespad=0.0, fontsize=18, ncol=3
    )
    plt.xscale("log")
    plt.xlabel("q-value cut-off", fontsize=16)
    plt.ylabel(plot_col, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(
        f"Extrapolated ratio for channels {icol} and {jcol}: " f"E-Coli", fontsize=18
    )
    plt.tight_layout()
    if output_file:
        file_format = output_file.split(".")[-1]
        plt.savefig(output_file, format=file_format, dpi=dpi)
    plt.show()
