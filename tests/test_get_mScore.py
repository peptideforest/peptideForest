import os

import pandas as pd

import add_mScore


def test_get_mScore():
    add_mScore.main("tests/_data/quant_data/", "tests/_data/raw_data/")
    test = pd.read_csv("tests/_data/raw_data/data_without_mscores_mscore.csv")
    os.remove("tests/_data/raw_data/data_without_mscores_mscore.csv")
    assert all(test["mScore"] == [8, 2, 3, 4, 5, 8, 9, 10, 0, 0])
