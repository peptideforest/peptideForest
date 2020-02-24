import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "engine": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Score_processed": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "Spectrum ID": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Is decoy": [0, 0, 1, 1, 1, 1, 0, 1, 0],
    }
)
df["Is decoy"] = df["Is decoy"].astype(bool)


def test_calc_delta_score_i():
    # [TRISTAN] missing values for to_decoy?
    df_test = peptide_forest.prep.calc_delta_score_i(df, 2, 1.5)
    assert all(df_test["delta_score_2"].isna())
    assert all(df_test["delta_score_2_delta_type"].isna())

    df_test = peptide_forest.prep.calc_delta_score_i(df, 2, 0.1)
    assert all(df_test.delta_score_2 == [-4, 0, 1, 0, -4, 1, 1, -4, 0])
    assert all(
        df_test.delta_score_2_delta_type.astype(int) == [1, 1, 3, 4, 4, 4, 1, 3, 1]
    )

    df_test = peptide_forest.prep.calc_delta_score_i(df, 3, 0.1)
    assert all(df_test.delta_score_3 == [0, 4, 5, 4, 0, 5, 5, 0, 4])
    assert all(
        df_test.delta_score_3_delta_type.astype(int) == [1, 1, 3, 4, 4, 4, 2, 4, 2]
    )
