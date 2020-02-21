import pandas as pd

import peptideForest

df = pd.DataFrame(
    {
        "Score_processed_test_eng": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "Sequence": ["AA", "AT", "AG", "AC", "CA", "CT", "CG", "CC", "TA", "TT"],
    }
)
df["Is decoy"] = df["Is decoy"].astype(bool)


def test_find_psms_to_keep():
    df_test = peptideForest.models.find_psms_to_keep(
        df, score_col="Score_processed_test_eng"
    )
    assert list(df_test["keep in"]) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_get_q_vals():
    # [TRISTAN] FDR calculation formula? 
    df_test_fdr_true = peptideForest.models.get_q_vals(
        df,
        score_col="Score_processed_test_eng",
        frac_tp=0.9,
        top_psm_only=False,
        initial_engine=None,
        get_fdr=True,
    )
    df_test_fdr_false = peptideForest.models.get_q_vals(
        df,
        score_col="Score_processed_test_eng",
        frac_tp=0.9,
        top_psm_only=False,
        initial_engine=None,
        get_fdr=False,
    )
    test_frames = [df_test_fdr_true, df_test_fdr_false]
    for frame in test_frames:
        assert frame.equals(
            frame.sort_values("Score_processed_test_eng", ascending=False)
        )
        assert list(frame.index) == [9, 2, 1, 0]
        assert all(frame["Decoy"].astype(bool) == frame["Is decoy"])
        assert all(frame["Decoy"].astype(bool) == ~frame["Target"].astype(bool))
        assert list(frame["FDR"]) == [0.0, 2.7, 1.35, 0.9]
    assert df_test_fdr_true["FDR"].equals(df_test_fdr_true["q-value"])
    assert list(df_test_fdr_false["q-value"]) == [0, 2.7, 2.7, 2.7]
