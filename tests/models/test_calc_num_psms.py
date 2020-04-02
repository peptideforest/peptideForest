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


def test_calc_num_psms():
    test_psms = peptide_forest.models.calc_num_psms(
        df=df, score_col="Score_processed_test_eng", q_cut=0.01, frac_tp=0.9
    )
    assert test_psms == 2


def test_get_top_targets():
    test = peptide_forest.models.get_top_targets(
        df=df, score_col="Score_processed_test_eng", q_cut=0.01, frac_tp=0.9
    )
    assert len(test) == 2
    assert list(test.loc[0]) == [10, 1, 0, "AA"]
    assert list(test.loc[9]) == [10, 4, 0, "TT"]


def test_get_train_set():
    test = peptide_forest.models.get_train_set(
        df=df,
        score_col="Score_processed_test_eng",
        q_cut=0.01,
        frac_tp=0.9,
        train_top_data=False,
        sample_frac=0.5,
    )
    test_top_data_only = peptide_forest.models.get_train_set(
        df=df,
        score_col="Score_processed_test_eng",
        q_cut=0.01,
        frac_tp=0.9,
        train_top_data=True,
        sample_frac=0.5,
    )
    test_top_data_only_frac_1 = peptide_forest.models.get_train_set(
        df=df,
        score_col="Score_processed_test_eng",
        q_cut=0.01,
        frac_tp=0.9,
        train_top_data=True,
        sample_frac=1,
    )
    assert len(test) == len(test_top_data_only) == 3
    assert len(test_top_data_only_frac_1) == 4
