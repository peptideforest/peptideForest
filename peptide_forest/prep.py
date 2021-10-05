import numpy as np
import multiprocessing as mp
from functools import partial

import pandas as pd

import peptide_forest.knowledge_base
from peptide_forest.tools import Timer


def _parallel_apply(df, func):
    chunks = np.array_split(df, mp.cpu_count() - 1)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        df = pd.concat(pool.map(func, chunks))

    return df


def add_stats(stats, df):
    df["_score_min"] = df.apply(lambda row: stats[row["Engine"]]["min_score"], axis=1)
    df["_score_max"] = df.apply(lambda row: stats[row["Engine"]]["max_score"], axis=1)

    return df


def check_mass_sanity(df):
    return (df.groupby(["Spectrum ID", "Sequence", "Charge", "Modifications"]).agg({"uCalc m/z": "nunique", "Exp m/z": "nunique"}) != 1).any(axis=0).any()


def calc_col_features(df):
    # Determine delta columns to calculate
    min_data = 0.7
    delta_columns = []
    size = df.groupby(["Engine", "Spectrum ID"]).size().droplevel(1)
    d2 = (size >= 2).groupby("Engine").agg("sum") / (size > 0).groupby("Engine").agg("sum")
    d3 = (size >= 3).groupby("Engine").agg("sum") / (size > 0).groupby("Engine").agg("sum")
    delta_columns += [f"delta_score_2_{col}" for col in d2[d2 >= min_data].index]
    delta_columns += [f"delta_score_3_{col}" for col in d3[d3 >= min_data].index]

    core_idx_cols = [
        "Spectrum Title",
        "Spectrum ID",
        "Sequence",
        "Modifications",
        "Is decoy",
        "Protein ID",
    ]
    value_cols = ["Score_processed"]
    remaining_idx_cols = [c for c in df.columns if not c in core_idx_cols + value_cols and c != "Engine"]

    # Pivot
    df = pd.pivot_table(
        df,
        index=core_idx_cols + remaining_idx_cols,
        values=value_cols,
        columns="Engine",)

    df.columns = ["_".join(t) for t in df.columns.to_list()]
    # Fill scores
    score_cols = [c for c in df.columns if c.startswith("Score_processed_")]
    df[score_cols] = df[score_cols].fillna(0.)
    df.reset_index(inplace=True)

    for delta_col in delta_columns:
        n = int(delta_col.split("_")[2])
        eng = delta_col[14:]
        max = df.groupby("Spectrum ID")[f"Score_processed_{eng}"].agg("max")
        nth_largest = df.groupby("Spectrum ID")[f"Score_processed_{eng}"].nlargest(n)
        nth_largest = nth_largest.loc[nth_largest.groupby(level=0).tail(1).index].droplevel(1)
        max - nth_largest


    return df

def calc_row_features(df):
    min_max_stats = df.groupby("Engine").agg({"Score": ["min", "max"]})["Score"]
    stats = {
        eng: {
            "min_score": 1e-30 if "omssa" in eng and min_score < 1e-30 else min_score,
            "max_score": max_score,
        }
        for eng, min_score, max_score in zip(
            min_max_stats.index, min_max_stats["min"], min_max_stats["max"]
        )
    }

    mp_add_stats = partial(add_stats, stats)
    df = _parallel_apply(df, mp_add_stats)
    df["Score_processed"] = df["Score"]
    df["Score_processed"].where(
        cond=~((df["_score_max"] < 1) & (df["Score"] <= df["_score_min"])),
        other=-np.log10(df["_score_min"]),
        inplace=True,
    )
    df["Score_processed"].where(
        cond=~((df["_score_max"] < 1) & (df["Score"] > df["_score_min"])),
        other=-np.log10(df["Score"]),
        inplace=True,
    )

    # Check for consistent masses
    if check_mass_sanity(df):
        raise ValueError("Recorded mass values are deviating across engines for identical matches. Rerunning ursgal is recommended.")

    df["Mass"] = (df["uCalc m/z"] - peptide_forest.knowledge_base.parameters["proton"]) * df["Charge"]
    df["dM"] = abs(df["uCalc m/z"] - df["Exp m/z"])

    # Trpysin specific cleavages
    df["enzN"] = (df["Sequence Pre AA"].str.contains(r"[RK-]") | df["Sequence Start"].isin(["1", "2"]))
    df["enzC"] = (df["Sequence Post AA"].str.contains(r"[-]") | df["Sequence"].str[-1].str.contains(r"[RK-]"))
    df["enzInt"] = df["Sequence"].str.count(r"[R|K]")

    # More stats
    df["PepLen"] = df["Sequence"].apply(len)
    df["CountProt"] = df["Protein ID"].str.split(r"<\|>|;").apply(set).apply(len)

    # Drop columns that are no longer relevant
    df.drop(columns=peptide_forest.knowledge_base.parameters["remove_after_row_features"], inplace=True)

    return df

def calc_features(df):
    with Timer("Computed features"):
        print("\nCalculating features:")

        with Timer("Computed row-level features"):
            print("Working on row-level features...")
            df = calc_row_features(df)

        with Timer("Computed column-level features"):
            print("Working on column-level features...")
            df = calc_col_features(df)
