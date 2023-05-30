"""Visualizing training progress and model performance."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

PALETTE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
]

ENGINE_REPL_DICT = {
    "comet_2020_01_4": "Comet 2020.01.4",
    "mascot_2_6_2": "Mascot 2.6.2",
    "msamanda_2_0_0_17442": "MSAmanda 2.0.0.17442",
    "msfragger_3_0": "MSFragger 3.0",
    "msgfplus_2021_03_22": "MSGF+ 2021.03.22",
    "omssa_2_1_9": "OMSSA 2.1.9",
    "peptide_forest": "PeptideForest",
}


def plot_model_performance(training_performance, title=None):
    """Plot model performance."""
    kpi_df = (
        pd.DataFrame(training_performance)
        .T.reset_index()
        .rename(
            columns={
                "index": "epoch",
                "mae": "Mean absolute error",
                "mse": "Mean squared error",
                "rmse": "Root mean squared error",
                "r2": "R2 score",
            }
        )
        .melt(id_vars=["epoch"], var_name="metric", value_name="value")
    )

    fig = (
        px.scatter(kpi_df, x="epoch", y="value", color="metric", range_y=[0, 1])
        .update_traces(mode="lines+markers")
        .update_layout(
            title="Model performance",
            xaxis_title="Epoch",
            yaxis_title="Metric value",
            legend_title="Metrics",
        )
    )
    if title is not None:
        fig.update_layout(title=title)

    fig.show()
    print()


def plot_q_value_curve(
    files: dict,
    palette: list = PALETTE,
    engine_repl_dict: dict = ENGINE_REPL_DICT,
    title=None,
):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df, index_col=0)
        df["dataset"] = dataset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    data = []
    q_val_cols = [c for c in df.columns if "q-value_" in c]
    for ds, grp in df.groupby("dataset"):
        for x in np.logspace(-4, -1, 100):
            for engine in q_val_cols:
                data.append(
                    [
                        ds,
                        x,
                        engine_repl_dict[engine.replace("q-value_", "")],
                        len(grp[grp[engine] <= x]),
                    ]
                )
    plt_df = pd.DataFrame(
        data, columns=["Dataset", "q-value threshold", "Engine", "n PSMs"]
    )
    sns.lineplot(
        data=plt_df, x="q-value threshold", y="n PSMs", hue="Engine", palette=palette
    )
    plt.title(title)
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(
        f"./plots/{title}_q_value_lines.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()


def plot_psms_at_qval_threshold(
    files: dict,
    palette: list = PALETTE,
    engine_repl_dict: dict = ENGINE_REPL_DICT,
    title=None,
):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df, index_col=0)
        df["dataset"] = dataset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    plt_df = pd.melt(
        df.groupby("dataset")[[c for c in df.columns if "top_target_" in c]]
        .agg("sum")
        .reset_index(),
        value_vars=[c for c in df.columns if "top_target_" in c],
        id_vars=["dataset"],
    )
    plt_df.columns = ["Dataset", "Engine", "nPSMs with q-val <= 1%"]
    plt_df["Engine"] = (
        plt_df["Engine"].str.replace("top_target_", "").replace(engine_repl_dict)
    )
    sns.barplot(
        data=plt_df,
        x="Dataset",
        y="nPSMs with q-val <= 1%",
        hue="Engine",
        palette=palette,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(f"./plots/{title}_npsms.png", dpi=400, bbox_inches="tight")
    plt.show()
