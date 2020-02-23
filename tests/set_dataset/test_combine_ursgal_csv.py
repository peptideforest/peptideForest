import pandas as pd
import pytest
import os

import peptide_forest

path_dict_small = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score",
    }
}

path_dict_medium = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score",
    },
    "tests/_data/omssa_2_1_9.csv": {"engine": "omssa", "score_col": "OMSSA:pvalue"},
}


def test_combine_ursgal_simple():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_small)
    assert df["engine"].unique() == "mascot"
    assert df["Mascot:Score"].equals(df["Score"])


def test_combine_ursgal_drops_proper_columns():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_small)
    for col in peptide_forest.knowledge_base.parameteres[
        "columns_to_be_removed_from_input_csvs"
    ]:
        assert col not in df.columns


def test_combine_ursgal_medium():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    assert list(df["engine"].unique()) == ["mascot", "omssa"]
