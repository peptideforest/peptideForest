import numpy as np
import pandas as pd
import pytest
import logging

from peptide_forest import prep, PeptideForest

path_dict_medium = {
    pytest._test_path
    / "_data"
    / "mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "mascot:score",
    },
    pytest._test_path
    / "_data"
    / "omssa_2_1_9.csv": {"engine": "omssa", "score_col": "omssa:pvalue"},
}

df_stats = pd.DataFrame(
    {
        "search_engine": [
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
        "score": [1, 2, 3, 10, 11, 12, 100, 100, 0, 42],
    }
)

df_mass = pd.DataFrame(
    {
        "raw_data_location": [
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
        ],
        "sequence": [
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
        "spectrum_id": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "charge": [1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "ucalc_mz": [1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        "exp_mz": [1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
        "modifications": [
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

df_mass2 = pd.DataFrame(
    {
        "raw_data_location": [
            "test1.raw",
            "test2.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
            "test1.raw",
        ],
        "sequence": [
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
        "spectrum_id": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "charge": [1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        "ucalc_mz": [1, 2, 1, 1, 1, 1, 2, 1, 1, 1],
        "exp_mz": [1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
        "modifications": [
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
        "search_engine": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "score_processed_A": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "score_processed_B": [1, 5, 6, 15, 11, 16, 26, 21, 25],
        "spectrum_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "is_decoy": [0, 0, 1, 1, 1, 1, 0, 1, 0],
    }
)
df_deltas["is_decoy"] = df_deltas["is_decoy"].convert_dtypes()

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
    stats = prep.get_stats(df_stats)
    df_test = prep.add_stats(stats, df_stats)
    min_vals = {"A": 1.0, "ASDFomssa_1_2_3": 1e-30, "B": 10.0, "C": 100.0}
    max_vals = {"A": 3, "ASDFomssa_1_2_3": 42, "B": 12, "C": 100}
    assert df_test.groupby("search_engine")["_score_min"].first().to_dict() == min_vals
    assert df_test.groupby("search_engine")["_score_max"].first().to_dict() == max_vals


def test_check_mass_sanity():
    assert prep.check_mass_sanity(df_mass) == True
    assert prep.check_mass_sanity(df_mass.drop(index=[6, 7, 8])) == False


def test_check_mass_sanity_different_raw_data_location():
    assert prep.check_mass_sanity(df_mass2) == True
    assert prep.check_mass_sanity(df_mass2.drop(index=[6, 7, 8])) == False


def test_calc_delta():
    delta_cols = [
        "delta_score_2_A",
        "delta_score_3_A",
        "delta_score_2_B",
        "delta_score_3_B",
    ]
    df_test = prep._parallel_calc_delta(df_deltas, delta_cols)
    assert set(df_deltas.columns).union(set(delta_cols)) == set(
        df_test.columns.to_list()
    )
    assert np.array_equal(df_test[delta_cols].values, deltas)


def test_get_stats():
    # Test easy correct stats for non-omssa engines
    stats = prep.get_stats(df_stats)
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
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_medium.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    df_test = prep.calc_row_features(pf.input_df)
    assert (
        len(
            set(df_test.columns).difference(
                {
                    "charge",
                    "raw_data_location",
                    "accuracy_ppm",
                    "comments",
                    "search_engine",
                    "is_decoy",
                    "modifications",
                    "protein_id",
                    "sequence",
                    "spectrum_id",
                    "spectrum_title",
                    "score_processed",
                    "mass",
                    "dm",
                    "enz_n",
                    "enz_c",
                    "enz_int",
                    "pep_len",
                    "count_prot",
                }
            )
        )
        == 0
    )
    assert all(df_test["pep_len"] == [9, 5, 5, 5, 7])
    assert all(df_test["count_prot"] == [1, 2, 2, 1, 1])


def test_col_features():
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_medium.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    df_test = prep.calc_row_features(pf.input_df)
    df_test = prep.calc_col_features(df_test, min_data=0.2)
    assert (
        len(
            set(df_test.columns).difference(
                {
                    "spectrum_title",
                    "spectrum_id",
                    "sequence",
                    "modifications",
                    "is_decoy",
                    "protein_id",
                    "charge",
                    "comments",
                    "mass",
                    "dm",
                    "enz_n",
                    "enz_c",
                    "enz_int",
                    "pep_len",
                    "count_prot",
                    "score_processed_mascot_2_6_2",
                    "score_processed_omssa_2_1_9",
                    "delta_score_2_omssa_2_1_9",
                    "raw_data_location",
                    "accuracy_ppm",
                }
            )
        )
        == 0
    )
    assert all(df_test["score_processed_mascot_2_6_2"] == [0.0, 0.0, 0.0, 0.0, 20.0])
    assert all(df_test["score_processed_omssa_2_1_9"] == [30.0, 29.0, 20.0, 10.0, 0.0])
    assert all(df_test["delta_score_2_omssa_2_1_9"] == [1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "feature_cols, expected, warning, message",
    [
        (
            ["XYZ", "charge", "accuracy_ppm"],
            ["charge", "accuracy_ppm"],
            True,
            "Could not find column XYZ in selected data. Dropped column",
        ),
        (["charge", "accuracy_ppm"], ["charge", "accuracy_ppm"], False, ""),
        (
            ["XYZ", "ABC", "QWE"],
            ["charge", "accuracy_ppm"],
            True,
            "Feature coluns not found in data. Fallback to default feature columns.",
        ),
        # TODO: define default feature columns
    ]
)
def test_fallback_for_wrong_feature_columns(
    caplog, feature_cols, expected, warning, message
):
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_medium.json",
        output=None,
    )
    pf.params["feature_cols"] = feature_cols

    if warning:
        caplog.set_level(logging.WARNING)
        pf.prep_ursgal_csvs()

        assert message in caplog.text
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
    else:
        pf.prep_ursgal_csvs()

    assert pf.feature_cols == expected


def test_cutom_core_columns_fallback(caplog):
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_wrong_core_cols.json",
        output=None,
    )
    caplog.set_level(logging.WARNING)
    pf.prep_ursgal_csvs()

    assert "sequence_x not found in file column names" in caplog.text
    assert "sequence_y not found in file column names" in caplog.text
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[1].levelname == "WARNING"

    assert pf.sequence_col == "sequence"


def test_custom_core_columns_global_config():
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_core_cols_global_config.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    assert "sequence_x" in pf.input_df.columns
    assert "sequence_y" not in pf.input_df.columns
    assert "decoy_x" in pf.input_df.columns


def test_non_numeric_feature_columns(caplog):
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_medium.json",
        output=None,
    )
    pf.params["feature_cols"] = ["sequence", "charge", "accuracy_ppm"]

    caplog.set_level(logging.WARNING)
    pf.prep_ursgal_csvs()

    assert "Column sequence is not numeric, dropped column from feature columns" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"

    assert pf.feature_cols == ["charge", "accuracy_ppm"]


def test_core_column_missing_in_file(caplog):
    caplog.set_level(logging.WARNING)
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_corrupted_omssa.json",
        output=None,
    )
    pf.prep_ursgal_csvs()

    assert "Column sequence could not be found in file " in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"

    assert len(pf.input_df) == 2


def test_mass_sanity_cols_not_available(caplog):
    pf = PeptideForest(
        config_path=pytest._test_path / "_data" / "path_dict_medium.json",
        output=None,
    )
    pf.prep_ursgal_csvs()
    pf.input_df.drop(columns=["modifications"], inplace=True)

    caplog.set_level(logging.WARNING)
    pf.calc_features()

    assert "Column modifications cannot be found in data, skipped mass sanity check!" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"


def test_custom_pep_len_configuration():
    pass


def test_custom_count_prot_configuration():
    pass
