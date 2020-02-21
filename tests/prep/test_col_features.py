import pandas as pd

import peptideForest


def test_calc_delta_score_i():
    # [TRISTAN] missing values for to_decoy?
    df = pd.DataFrame(
        {
            "engine": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "Score_processed": [1, 5, 6, 15, 11, 16, 26, 21, 25],
            "Spectrum ID": ["1", "1", "1", "1", "1", "1", "1", "1", "1"],
            "Is decoy": ["0", "0", "1", "1", "1", "1", "0", "1", "0"],
        }
    )
    try:
        peptideForest.prep.calc_delta_score_i(df, 2, 0.7)
        assert False
    except:
        peptideForest.prep.calc_delta_score_i(df, 2, 0.1)
        peptideForest.prep.calc_delta_score_i(df, 3, 0.1)
    assert all(df.delta_score_2 == [-4, 0, 1, 0, -4, 1, 1, -4, 0])
    assert all(df.delta_score_2_to_decoy.astype(int) == [0, 0, 0, 1, 1, 1, 0, 0, 0])
    assert all(df.delta_score_3 == [0, 4, 5, 4, 0, 5, 5, 0, 4])
    assert all(df.delta_score_3_to_decoy.astype(int) == [0, 0, 0, 1, 1, 1, 1, 1, 1])
