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


def test_cleavage_aa():
    df = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df["enzN"] = df.apply(
        lambda x: peptideForest.prep.test_cleavage_aa(x["Sequence Pre AA"], x["Sequence Start"]),
        axis=1,
    )
    assert all(df[df["engine"] == "omssa"]["enzN"]) is True

def test_sequence_aa_c():
    df = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df["enzC"] = df.apply(
        lambda x: peptideForest.prep.test_sequence_aa_c(x["Sequence"][-1], x["Sequence Post AA"]),
        axis=1,
    )
    assert all(df[df["Comments"] == "test_sequence_aa_c"]) is True

def test_cound_missed_cleavages():
    df = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df['enzInt'] = df['Sequence'].str.count(r"[R|K]")
    assert df[df["Comments"] == "Two missed cleavages"]["enzInt"].unique() == 3

