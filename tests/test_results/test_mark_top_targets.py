import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "top_target_to_be_dropped": [1, 2, 3, 5, 5, 5, 7, 6, 7, 10],
        "q-value_a": [0.67, 0.73, 0.2, 0.61, 0.42, 0.23, 0.77, 0.04, 0.56, 0.6],
        "q-value_b": [0.28, 0.24, 0.61, 0.61, 0.63, 0.57, 0.54, 0.29, 0.25, 0.73],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    }
)


def test_mark_top_targets():
    df_test = peptide_forest.results.mark_top_targets(df=df, q_cut=0.5)
    assert "top_target_to_be_dropped" not in df_test.columns
    assert df_test[df_test["top_target_a"] == 1].index == 5
    assert all(df_test[df_test["top_target_b"] == 1].index == [1, 8])
