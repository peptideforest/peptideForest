import numpy as np
import pandas as pd

import peptide_forest

df = pd.DataFrame(
    {
        "engine": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "Score_processed_A": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "Score_processed_B": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "Spectrum ID": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Is decoy": [0, 0, 1, 1, 1, 1, 0, 1, 0],
    }
)
df["Is decoy"] = df["Is decoy"].convert_dtypes()

deltas = [
    [-24, -20, -24, -20],
    [-20, -16, -20, -16],
    [-19, -15, -19, -15],
    [-10, -6, -10, -6],
    [-14, -10, -14, -10],
    [-9, -5, -9, -5],
    [1, 5, 1, 5],
    [-4, 0, -4, 0],
    [0, 4, 0, 4],
]


def test_calc_delta_score_i():
    print("a")
    df_test = df.groupby("Spectrum ID").apply(
        peptide_forest.prep.calc_deltas,
        delta_lookup={
            "Score_processed_A": [
                {"column": "delta_2_A", "iloc": 1},
                {"column": "delta_3_A", "iloc": 2},
            ],
            "Score_processed_B": [
                {"column": "delta_2_B", "iloc": 1},
                {"column": "delta_3_B", "iloc": 2},
            ],
        },
    )
    assert df_test.columns.to_list() == [
        "engine",
        "Score_processed_A",
        "Score_processed_B",
        "Spectrum ID",
        "Is decoy",
        "delta_2_A",
        "delta_3_A",
        "delta_2_B",
        "delta_3_B",
    ]
    assert np.array_equal(df_test.iloc[:, -4:].values, deltas)
