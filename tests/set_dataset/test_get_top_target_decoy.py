import pandas as pd

import peptide_forest


def test_get_top_target_decoy():

    df = pd.DataFrame(
        {
            "Score": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "Is decoy": [0, 0, 0, 1, 1, 1, 0, 1, 0],
            "Comment": [
                "a",
                "b",
                "target for Spectrum ID 1 / no decoy",
                "d",
                "e",
                "decoy for Spectrum ID 2 / no target",
                "g",
                "decoy for Spectrum ID 3",
                "target for Spectrum ID 3",
            ],
        }
    )
    df = peptide_forest.setup_dataset.get_top_target_decoy(df, score_col="Score")

    assert len(df) == 4
    assert set(df["Score"]) == {3, 6, 8, 9}
