import numpy as np
import pandas as pd
import pytest

import peptide_forest

path_dict_medium = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score",
    },
    "tests/_data/omssa_2_1_9.csv": {"engine": "omssa", "score_col": "OMSSA:pvalue"},
}

df_stats = pd.DataFrame(
    {
        "Search Engine": [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "C",
            "C",
            "ASDFomssa_1_2_3",
            "ASDFomssa_1_2_3",
        ],
        "Score": [1, 2, 3, 10, 11, 12, 100, 100, 0, 42],
    }
)

df_mass = pd.DataFrame(
    {
        "Sequence": [
            "AAA",
            "AAA",
            "AAA",
            "AAB",
            "AAB",
            "AAB",
            "AAB",
            "AAB",
            "AAB",
            "AAB",
        ],
        "Spectrum ID": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "Charge": [1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "uCalc m/z": [1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        "Exp m/z": [1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "Modifications": [
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAA",
            "AAB",
        ],
    }
)

df_deltas = pd.DataFrame(
    {
        "Search Engine": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "Score_processed_A": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "Score_processed_B": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "Spectrum ID": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Is decoy": [0, 0, 1, 1, 1, 1, 0, 1, 0],
    }
)
df_deltas["Is decoy"] = df_deltas["Is decoy"].convert_dtypes()

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


def test_add_stats():
    # Test correct values are inserted mapped to whole df
    stats = peptide_forest.prep.get_stats(df_stats)
    df_test = peptide_forest.prep.add_stats(stats, df_stats)
    min_vals = {"A": 1.0, "ASDFomssa_1_2_3": 1e-30, "B": 10.0, "C": 100.0}
    max_vals = {"A": 3, "ASDFomssa_1_2_3": 42, "B": 12, "C": 100}
    assert df_test.groupby("Search Engine")["_score_min"].first().to_dict() == min_vals
    assert df_test.groupby("Search Engine")["_score_max"].first().to_dict() == max_vals


def test_check_mass_sanity():
    assert peptide_forest.prep.check_mass_sanity(df_mass) == True
    assert peptide_forest.prep.check_mass_sanity(df_mass.drop(index=[6, 7, 8])) == False


def test_calc_delta():
    delta_cols = [
        "delta_score_2_A",
        "delta_score_3_A",
        "delta_score_2_B",
        "delta_score_3_B",
    ]
    df_test = peptide_forest.prep._parallel_calc_delta(df_deltas, delta_cols)
    assert set(df_deltas.columns).union(set(delta_cols)) == set(
        df_test.columns.to_list()
    )
    assert np.array_equal(df_test[delta_cols].values, deltas)


def test_get_stats():
    # Test easy correct stats for non-omssa engines
    stats = peptide_forest.prep.get_stats(df_stats)
    assert stats["A"]["min_score"] == 1
    assert stats["A"]["max_score"] == 3
    assert stats["B"]["min_score"] == 10
    assert stats["B"]["max_score"] == 12
    assert stats["C"]["min_score"] == 100
    assert stats["C"]["max_score"] == 100

    # Test for omssa crazyness
    assert stats["ASDFomssa_1_2_3"]["min_score"] == 1e-30
    assert stats["ASDFomssa_1_2_3"]["max_score"] == 42


@pytest.mark.filterwarnings("ignore")
def test_row_features():
    pf = peptide_forest.PeptideForest(
        initial_engine="omssa",
        ursgal_path_dict="tests/_data/path_dict_medium.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    df_test = peptide_forest.prep.calc_row_features(pf.input_df)
    assert (
        len(
            set(df_test.columns).difference(
                {
                    "Charge",
                    "Raw data location",
                    "Accuracy (ppm)",
                    "Comments",
                    "Search Engine",
                    "Is decoy",
                    "Modifications",
                    "Protein ID",
                    "Sequence",
                    "Spectrum ID",
                    "Spectrum Title",
                    "Score_processed",
                    "Mass",
                    "dM",
                    "enzN",
                    "enzC",
                    "enzInt",
                    "PepLen",
                    "CountProt",
                }
            )
        )
        == 0
    )
    assert all(df_test["PepLen"] == [9, 5, 5, 5, 7])
    assert all(df_test["CountProt"] == [1, 2, 2, 1, 1])


def test_col_features():
    pf = peptide_forest.PeptideForest(
        initial_engine="omssa",
        ursgal_path_dict="tests/_data/path_dict_medium.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    df_test = peptide_forest.prep.calc_row_features(pf.input_df)
    df_test = peptide_forest.prep.calc_col_features(df_test, min_data=0.2)
    assert (
        len(
            set(df_test.columns).difference(
                {
                    "Spectrum Title",
                    "Spectrum ID",
                    "Sequence",
                    "Modifications",
                    "Is decoy",
                    "Protein ID",
                    "Charge",
                    "Comments",
                    "Mass",
                    "dM",
                    "enzN",
                    "enzC",
                    "enzInt",
                    "PepLen",
                    "CountProt",
                    "Score_processed_mascot_2_6_2",
                    "Score_processed_omssa_2_1_9",
                    "delta_score_2_omssa_2_1_9",
                    "reported_by_mascot_2_6_2",
                    "reported_by_omssa_2_1_9",
                    "Raw data location",
                    "Accuracy (ppm)",
                }
            )
        )
        == 0
    )
    assert all(df_test["Score_processed_mascot_2_6_2"] == [0.0, 0.0, 0.0, 0.0, 20.0])
    assert all(df_test["Score_processed_omssa_2_1_9"] == [30.0, 29.0, 20.0, 10.0, 0.0])
    assert all(df_test["delta_score_2_omssa_2_1_9"] == [1.0, 0.0, 0.0, 0.0, 0.0])
