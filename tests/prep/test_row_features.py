import pandas as pd
import numpy as np
import pytest
import os

import peptideForest

path_dict_medium = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score"
    },
    "tests/_data/omssa_2_1_9.csv": {
        "engine": "omssa",
        "score_col": "OMSSA:pvalue"
    }
}


def test_row_features():
    max_charge = 7
    df = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df = peptideForest.prep.row_features(df, "C", proton=1, max_charge=max_charge)

    assert df["Mass"].unique() == 3000
    assert df["Accuracy (ppm)"].equals(df["delta m/z"])
    assert df["ln(abs delta m/z + 1)"].min() == 0 
    assert df[df["Comments"] == "CountProt equals 2"]['CountProt'].unique() == 2

    for charge in range(1, max_charge - 1):
        assert f'Charge{int(charge)}' in df.columns
    assert f'>Charge{max_charge}' in df.columns

    for index, row in df.iterrows():
        if row['Charge'] >= max_charge:
            assert row[f'>Charge{max_charge}'] == 1
        else:
            assert row[f"Charge{row['Charge']}"] == 1