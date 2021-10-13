import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "Score_processed_a": [85, 55, 41, 28, 76, 57, 8, 27, 8, 64],
        "Score_processed_b": [37, 76, 65, 4, 85, 17, 48, 14, 77, 66],
        "Score_processed_c": [97, 85, 45, 83, 97, 18, 28, 42, 71, 12],
        "Remainder": ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"],
    }
)


def test_mark_top_targets():
    df_test = peptide_forest.results.get_ranks(df=df)
    assert all(df_test["rank_a"] == [1, 5, 6, 7, 2, 4, 9, 8, 10, 3])
    assert all(df_test["rank_b"] == [7, 3, 5, 10, 1, 8, 6, 9, 2, 4])
    assert all(df_test["rank_c"] == [1, 3, 6, 4, 2, 9, 8, 7, 5, 10])
