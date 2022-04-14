import pandas as pd

from peptide_forest import training

df = pd.DataFrame(
    {
        "score_processed_test_eng": [10, 7, 8, 8, 8, 8, 7, 6, 7, 10, 5, 5, 4],
        "spectrum_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5],
        "is_decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        "sequence": [
            "AA",
            "AT",
            "AG",
            "AC",
            "CA",
            "CT",
            "CG",
            "CC",
            "TA",
            "TT",
            "AA",
            "AB",
            "AC",
        ],
    }
)
df["is_decoy"] = df["is_decoy"].astype(bool)


def test_find_psms_to_keep():
    df_test = training.find_psms_to_keep(df, score_col="score_processed_test_eng")
    assert df_test.index.tolist() == [0, 1, 2, 9, 10, 11, 12]


def test_calc_num_psms():
    test_psms = training.calc_num_psms(
        df=df, score_col="score_processed_test_eng", q_cut=0.5, sensitivity=0.9
    )
    assert test_psms == 3


def test_get_q_vals():
    df_test_fdr_true = training.calc_q_vals(
        df,
        score_col="score_processed_test_eng",
        sensitivity=0.9,
        top_psm_only=False,
        init_score_col=None,
        get_fdr=True,
    )
    df_test_fdr_false = training.calc_q_vals(
        df,
        score_col="score_processed_test_eng",
        sensitivity=0.9,
        top_psm_only=False,
        init_score_col=None,
        get_fdr=False,
    )
    for frame in [df_test_fdr_true, df_test_fdr_false]:
        assert frame.equals(
            frame.sort_values("score_processed_test_eng", ascending=False)
        )
        assert set(frame.index) == set([0, 9, 1, 2, 10, 11, 12])
        assert all(frame["decoy"].astype(bool) == frame["is_decoy"])
        assert all(frame["decoy"].astype(bool) == ~frame["target"].astype(bool))
        assert list(frame["fdr"]) == [
            0.0,
            0.0,
            0.3,
            0.225,
            0.18,
            0.15,
            0.2571428571428572,
        ]
    assert df_test_fdr_true["fdr"].equals(df_test_fdr_true["q-value"])
    assert list(df_test_fdr_false["q-value"]) == [
        0.0,
        0.0,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
    ]
