import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "Score_processed_initial": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Score_processed_a": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Score_processed_b": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "Sequence": ["AA", "AT", "AG", "AC", "CA", "CT", "CG", "CC", "TA", "TT"],
    }
)
df["Is decoy"] = df["Is decoy"].astype(bool)


def test_calc_all_final_q_vals():

    df_test = peptide_forest.results.calc_all_final_q_vals(
        df=df, frac_tp=0.9, top_psm_only=True, initial_engine="Score_processed_initial"
    )
    df_initial = peptide_forest.models.get_q_vals(
        df=df,
        score_col="Score_processed_initial",
        frac_tp=0.9,
        top_psm_only=True,
        initial_engine="Score_processed_initial",
    )
    df_a = peptide_forest.models.get_q_vals(
        df=df,
        score_col="Score_processed_a",
        frac_tp=0.9,
        top_psm_only=True,
        initial_engine="Score_processed_initial",
    )
    df_b = peptide_forest.models.get_q_vals(
        df=df,
        score_col="Score_processed_b",
        frac_tp=0.9,
        top_psm_only=True,
        initial_engine="Score_processed_initial",
    )
    assert len(df_test.columns) == 9
    assert df_test[df_test["q-value_initial"] != 1]["q-value_initial"].equals(
        df_initial["q-value"].sort_index()
    )
    assert df_test[df_test["q-value_a"] != 1]["q-value_a"].equals(
        df_a["q-value"].sort_index()
    )
    assert df_test[df_test["q-value_b"] != 1]["q-value_b"].equals(
        df_b["q-value"].sort_index()
    )
