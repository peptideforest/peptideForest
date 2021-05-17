import peptide_forest

path_dict_medium = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score",
    },
    "tests/_data/omssa_2_1_9.csv": {"engine": "omssa", "score_col": "OMSSA:pvalue"},
}


def test_row_features():
    max_charge = 7
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df = peptide_forest.prep.row_features(
        df,
        "C",
        proton=1,
        max_charge=max_charge,
        features={"row_features": set([]), "transformed_features": set([])},
    )[0]

    assert df["Mass"].unique() == 3000
    assert df[df["Comments"] == "CountProt equals 2"]["CountProt"].unique() == 2
    assert all(df["Charge"] == [3, 3, 3, 3, 3, 3, 1, 2, 3, 3, 3, 10])
