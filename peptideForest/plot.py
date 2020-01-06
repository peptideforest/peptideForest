from peptideForest import results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import itertools


def plot_num_psms_by_method(df, methods, output_file, dpi, show_plot):
    """
    Plot the number of PSMs for each method with available results.
    Args:
        df (pd.DataFrame): dataframe containing results from search engines and ML training
        methods (List): list of methods to use, if None, use all methods
        output_file (str): path to save new dataframe to
        dpi (int): plotting resolution
        show_plot (bool): display plot
    """
    # Get the number of PSMs for each method
    df_num_psms = results.get_num_psms_by_method(df, methods=methods)

    # Initiate the plot
    plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
    ax = sns.barplot(x="method", y="n_psms", data=df_num_psms)

    # Print values [TRISTAN] now fixed
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
    df_training, q_val_cut, initial_engine, methods, output_file, show_plot, dpi
):
    """
    Plot the number of PSMs for each method with available results.
    Args:
        df_training (pd.DataFrame): dataframe containing results from search engines and ML training
        q_val_cut (float): q-value used to identify top targets
        initial_engine (str): name of initial engine
        methods (List): list of methods to use, if None, use all methods
        output_file (str): path to save new dataframe to
        show_plot (bool): display plot
        dpi (int): plotting resolution
    """
    # Get the q-value list
    if q_val_cut is None:
        q_val_cut = sorted(
            [float(f"{i}e-{j}") for i in np.arange(1, 10) for j in np.arange(4, 1, -1)]
        ) + [1e-1]

    # Get dataframe for number of PSMs as a function of q-value
    df_num_psms_q = results.get_num_psms_against_q_cut(
        df_training, methods=methods, q_val_cut=q_val_cut, initial_engine=initial_engine
    )

    # Markers to be used for plotting
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

    # Initiate the plot
    plt.figure(num=None, figsize=(13, 8), dpi=dpi, facecolor="w", edgecolor="k")
    plt.title(f"Number of PSMs identified against q-value used as cut-off", fontsize=18)
    x = q_val_cut
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

    # # re-calculate the top-targets using q_cut = 1% [TRISTAN] ????
    # cols = [c for c in df if "top_target_" in c]
    # df = df.drop(cols, axis=1)
    # df = get_top_targets(df, q_val_cut=0.01)
    # # ^---- why ?


def plot_ranks(df, x_name, y_name, use_top_psm, n_psms, output_file, show_plot, dpi):
    """
    Plot ranks of PSMs from two methods against each other.
    Args:
        df (pd.DataFrame): dataframe containing ranks from different methods
        x_name (str): name of method for the x-axis
        y_name (str): name of method for the y-axis
        use_top_psm (bool): use the top PSMs for each spectrum (True, default), otherwise use all PSMs
        n_psms (int): number of psms to plot results for, in multiples of the number with q-value<1%
        output_file (str): name of output file for image. If None (default), don't save the image
        show_plot (bool): display plots
        dpi (int): plotting resolution

    """
    # Initiate the plot
    plt.figure(num=None, figsize=(13, 13), dpi=dpi, facecolor="w", edgecolor="k")

    # Names of columns to use
    col_x = f"rank_{x_name}"
    tt_x = f"top_target_{x_name}"

    col_y = f"rank_{y_name}"
    tt_y = f"top_target_{y_name}"

    if use_top_psm:
        df_plot = df.sort_values(
            f"Score_processed_{y_name}", ascending=False
        ).drop_duplicates("Spectrum ID")
        df_plot = results.get_ranks(df_plot, from_scores=True)
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
    lgnd.legendHandles[0]._legmarker.set_markersize(12)
    lgnd.legendHandles[1]._legmarker.set_markersize(12)
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

    plt.close()


def all(
    df_training,
    classifier,
    methods,
    output_file,
    all_engines_truncated,
    initial_engine,
    plot_prefix,
    plot_dir,
    show_plot,
    dpi,
):
    """
    Main function to plot.
    Args:
        df_training (pd.DataFrame): dataframe containing results from search engines and ML training
        classifier (str): q-value used to identify top targets
        methods (List): list of methods to use, if None, use all methods
        all_engines_truncated (List): List containing truncated engine names
        initial_engine (str): name of initial engine
        output_file (str): path to save new dataframe to
        dpi (int): plotting resolution

    """
    plot_num_psms_by_method(
        df_training,
        output_file=os.path.join(
            plot_dir, f"{plot_prefix}_PSMs_by" f"" f"" f"_method.pdf"
        ),
        show_plot=show_plot,
        methods=methods,
        dpi=dpi,
    )

    plot_num_psms_against_q(
        df_training,
        q_val_cut=None,
        methods=methods,
        output_file=os.path.join(plot_dir, f"{plot_prefix}_num_psms_vs_q.pdf"),
        show_plot=show_plot,
        dpi=dpi,
        initial_engine=initial_engine,
    )

    # [TRISTAN] remove show_plots; add use_top_psm/n_psms
    for e1, e2 in itertools.permutations(all_engines_truncated + [classifier], 2):
        plot_ranks(
            df_training,
            e1,
            e2,
            use_top_psm=True,
            n_psms=3,
            output_file=os.path.join(plot_dir, f"{plot_prefix}_{e1}_vs_{e2}.pdf"),
            show_plot=False,
            dpi=dpi,
        )
