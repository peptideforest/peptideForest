import pandas as pd
import numpy as np

import peptide_forest

global df
df = pd.DataFrame(
    {
        "engine": ["A", "B", "A", "B", "B"],
        "Score_processed": [1, 2, 3, 4, 5],
        "delta_score_2": [11, 22, 33, 44, 55],
        "delta_score_3": [111, 222, 333, 444, 555],
        "delta_score_2_to_decoy": [1, 1, 1, 1, 1],
        "delta_score_3_to_decoy": [0, 0, 0, 0, 0],
        "Mass": [1.1, 1.2, 1.3, 1.4, 1.5],
        "delta m/z": [2.1, 2.2, 2.3, 2.4, 2.5],
        "abs delta m/z": [3.1, 3.2, 3.3, 3.4, 3.5],
        "ln abs delta m/z": [4.1, 4.2, 4.3, 4.4, 4.5],
        "Spectrum ID": [1, 1, 2, 2, 1],
        "Sequence": ["AAAA", "AAAA", "BBBB", "BBCB", "CCCC"],
        "Modifications": ["moda", "moda", "modc", "modd", "mode"],
        "Protein ID": [1, 1, 2, 2, 1],
        "Is decoy": [0, 0, 1, 0, 1],
        "Remainder": ["a", "b", "c", "d", "e"],
    }
)
df.loc[5] = ["C", 1] + 4 * [np.nan] + [100, 100, 100, 100] + 6 * [np.nan]


def test_combine_engine_data():
    # [TRISTAN] missing values for to_decoy?
    # Check successful merging
    start_cols = set(df.columns)
    df_test = peptide_forest.prep.combine_engine_data(df, feature_cols=[])
    new_cols1 = set(df_test.columns) - start_cols

    # Check if correct columns are created and all-NaN columns are dropped

    ref_new_cols1 = {
        "Score_processed_A",
        "Score_processed_B",
        "Score_processed_C",
        "delta_score_2_A",
        "delta_score_2_B",
        "delta_score_2_to_decoy_A",
        "delta_score_2_to_decoy_B",
        "delta_score_3_A",
        "delta_score_3_B",
        "delta_score_3_to_decoy_A",
        "delta_score_3_to_decoy_B",
    }
    assert new_cols1 == ref_new_cols1

    # Check successful averaging
    avg_cols = ["Mass", "delta m/z", "abs delta m/z", "ln abs delta m/z"]
    assert all(df_test.round(2).loc[0, avg_cols] == [1.15, 2.15, 3.15, 4.15])
    assert all(df_test.round(2).loc[1, avg_cols] == [1.3, 2.3, 3.3, 4.3])
