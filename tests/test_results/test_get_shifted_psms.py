import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "top_target_ref": [0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        "top_target_new": [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        "rank_ref": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "rank_new": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Spectrum ID": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "Sequence": ["AA", "AA", "AA", "AA", "AA", "TT", "TT", "TT", "TT", "TT"],
        "Modifications": [
            "moda",
            "moda",
            "moda",
            "moda",
            "moda",
            "modb",
            "modb",
            "modb",
            "modb",
            "modb",
        ],
        "Protein ID": [1, 1, 3, 1, 1, 2, 2, 2, 4, 2],
        "Score_processed_ref": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Score_processed_new": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    }
)
bool_cols = ["top_target_ref", "top_target_new"]
df[bool_cols] = df[bool_cols].astype(bool)


def test_get_shifted_psms():
    # [TRISTAN] Note: the columns up and down rank refer to the top target state and NOT the rank... rename?
    df_new_top_targets, df_old_top_targets = peptide_forest.results.get_shifted_psms(
        df=df, x_name="ref", y_name="new", n_return=None
    )
    # Check new top targets
    assert all(df_new_top_targets.iloc[0, [0, -1]] == [4, False])
    assert all(df_new_top_targets.iloc[1, [0, -1]] == [3, False])
    # Check old top targets
    assert all(df_old_top_targets.iloc[0, [0, -1, -2]] == [2, True, True])
    assert all(df_old_top_targets.iloc[1, [0, -1, -2]] == [6, False, False])
    assert all(df_old_top_targets.iloc[2, [0, -1, -2]] == [8, True, True])
