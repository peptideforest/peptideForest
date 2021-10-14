import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "Score_processed_test_eng": [10, 7, 8, 8, 9, 5, 7, 6, 7, 10],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        "Sequence": ["AA", "AT", "AG", "AC", "CA", "CT", "CG", "CC", "TA", "TT"],
    }
)
df["Is decoy"] = df["Is decoy"].astype(bool)


def test_find_psms_to_keep():
    df_test = peptide_forest.training.find_psms_to_keep(
        df, score_col="Score_processed_test_eng"
    )
    # TODO: double check
    assert df_test.index.tolist() == [0, 1, 2, 3, 4, 5, 9]


def test_calc_num_psms():
    test_psms = peptide_forest.training.calc_num_psms(
        df=df, score_col="Score_processed_test_eng", q_cut=0.01, sensitivity=0.9
    )
    assert test_psms == 2


def test_get_q_vals():
    df_test_fdr_true = peptide_forest.training.calc_q_vals(
        df,
        score_col="Score_processed_test_eng",
        sensitivity=0.9,
        top_psm_only=False,
        init_score_col=None,
        get_fdr=True,
    )
    df_test_fdr_false = peptide_forest.training.calc_q_vals(
        df,
        score_col="Score_processed_test_eng",
        sensitivity=0.9,
        top_psm_only=False,
        init_score_col=None,
        get_fdr=False,
    )
    for frame in [df_test_fdr_true, df_test_fdr_false]:
        assert frame.equals(
            frame.sort_values("Score_processed_test_eng", ascending=False)
        )
        # assert list(frame.index) == [9, 6, 3, 2]
        assert set(frame.index) == set([0, 9, 4, 2, 3, 1, 5])
        assert all(frame["Decoy"].astype(bool) == frame["Is decoy"])
        assert all(frame["Decoy"].astype(bool) == ~frame["Target"].astype(bool))
        assert list(frame["FDR"]) == [0.0, 0.0, 0.45, 0.9, 0.6, 0.45, 0.36]
    assert df_test_fdr_true["FDR"].equals(df_test_fdr_true["q-value"])
    assert list(df_test_fdr_false["q-value"]) == [0.0, 0.0, 0.45, 0.9, 0.9, 0.9, 0.9]
