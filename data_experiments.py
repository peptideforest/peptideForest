import itertools
import multiprocessing as mp
from uuid import uuid4

import pandas as pd
import plotly.express as px

import peptide_forest
from peptide_forest.pf_config import PFConfig
from peptide_forest.visualization import plot_q_value_curve, plot_psms_at_qval_threshold

CONFIGS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, 9],
    "q_cut": [0.001, 0.01, 0.1],
    "n_spectra": [1000, 2500, 5000, 10000],
}


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    combinations = itertools.product(*CONFIGS.values())
    results = []
    for conf in combinations:
        output = f"./docker_test_data/{uuid4()}_output.csv"
        pf = peptide_forest.PeptideForest(
            config_path="./docker_test_data/config_local.json",  # args.c
            output=output,  # args.o,
            memory_limit=None,  # args.m,
            max_mp_count=1,  # args.mp_limit,
        )
        pf.config = PFConfig(
            {
                "n_estimators": {"value": conf[0]},
                "max_depth": {"value": conf[1]},
                "q_cut": {"value": conf[2]},
                "n_spectra": {"value": conf[3]},
                "n_folds": {"value": 3},
                "n_train": {"value": 1},
                "n_jobs": {"value": 7},
            }
        )
        pf.initial_config = pf.config.copy()
        pf.boost()
        df = pd.read_csv(output)
        plt_df = df[[c for c in df.columns if c.startswith("q-value")]]
        n_psms_10pct = plt_df.copy().loc[
            plt_df["q-value_peptide_forest"] <= 0.1, "q-value_peptide_forest"
        ].count()
        n_psms_1pct = plt_df.copy().loc[
            plt_df["q-value_peptide_forest"] <= 0.01, "q-value_peptide_forest"
        ].count()
        n_psms_0_1pct = plt_df.copy().loc[
            plt_df["q-value_peptide_forest"] <= 0.001, "q-value_peptide_forest"
        ].count()
        n_psms_0_01pct = plt_df.copy().loc[
            plt_df["q-value_peptide_forest"] <= 0.0001, "q-value_peptide_forest"
        ].count()
        results.append(
            {
                "n_estimators": conf[0],
                "max_depth": conf[1],
                "q_cut": conf[2],
                "n_spectra": conf[3],
                "n_psms_10%": n_psms_10pct,
                "n_psms_1%": n_psms_1pct,
                "n_psms_0.1%": n_psms_0_1pct,
                "n_psms_0.01%": n_psms_0_01pct,
                "complexity": conf[0] * conf[1],
            }
        )
        print(results[-1])

    results_df = pd.DataFrame(results)
    print()
