import pandas as pd

import peptide_forest

df_q_vals = pd.DataFrame(
    {
        "Score_processed_initial": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Score_processed_a": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Score_processed_b": [1, 2, 3, 15, 5, 5, 7, 6, 15, 10],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "Sequence": ["AA", "AT", "AG", "AC", "CA", "CT", "CG", "CC", "TA", "TT"],
    }
)
df_q_vals["Is decoy"] = df_q_vals["Is decoy"].astype(bool)

df_rank = pd.DataFrame(
    {
        "Score_processed_a": [85, 55, 41, 28, 76, 57, 8, 27, 8, 64],
        "Score_processed_b": [37, 76, 65, 4, 85, 17, 48, 14, 77, 66],
        "Score_processed_c": [97, 85, 45, 83, 97, 18, 28, 42, 71, 12],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "Remainder": ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"],
        "Sequence": ["AA", "AT", "AG", "AC", "CA", "CT", "CG", "CC", "TA", "TT"],
    }
)


def calc_all_final_q_vals():

    df_test = peptide_forest.results.process_final(
        df=df_q_vals, sensitivity=0.9, init_eng="Score_processed_initial", q_cut=0.01
    )
    df_initial = peptide_forest.training.calc_q_vals(
        df=df_q_vals,
        score_col="Score_processed_initial",
        sensitivity=0.9,
        top_psm_only=True,
        init_score_col="Score_processed_initial",
        get_fdr=True,
    )
    df_a = peptide_forest.training.calc_q_vals(
        df=df_q_vals,
        score_col="Score_processed_a",
        sensitivity=0.9,
        top_psm_only=True,
        init_score_col="Score_processed_initial",
        get_fdr=True,
    )
    df_b = peptide_forest.training.calc_q_vals(
        df=df_q_vals,
        score_col="Score_processed_b",
        sensitivity=0.9,
        top_psm_only=True,
        init_score_col="Score_processed_initial",
        get_fdr=True,
    )
    assert len(df_test.columns) == 15
    assert df_test[df_test["q-value_initial"] != 1]["q-value_initial"].equals(
        df_initial["q-value"].sort_index()
    )
    assert df_test[df_test["q-value_a"] != 1]["q-value_a"].equals(
        df_a["q-value"].sort_index()
    )
    assert df_test[df_test["q-value_b"] != 1]["q-value_b"].equals(
        df_b["q-value"].sort_index()
    )


def get_ranks():
    df_test = peptide_forest.results.process_final(
        df=df_rank, sensitivity=0.9, init_eng="Score_processed_initial", q_cut=0.01
    )
    assert all(df_test["rank_a"] == [1, 5, 6, 7, 2, 4, 9, 8, 10, 3])
    assert all(df_test["rank_b"] == [7, 3, 5, 10, 1, 8, 6, 9, 2, 4])
    assert all(df_test["rank_c"] == [1, 3, 6, 4, 2, 9, 8, 7, 5, 10])


def mark_top_targets():
    df_test = peptide_forest.results.process_final(
        df=df_q_vals, sensitivity=0.9, init_eng="Score_processed_initial", q_cut=0.5
    )
    assert df_test[df_test["top_target_a"] == 1].index == 9
    assert all(df_test[df_test["top_target_b"] == 1].index == [3, 8, 9])


def test_process_final():
    calc_all_final_q_vals()
    get_ranks()
    mark_top_targets()
