import peptide_forest

path_dict_medium = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score",
    },
    "tests/_data/omssa_2_1_9.csv": {"engine": "omssa", "score_col": "OMSSA:pvalue"},
}


def test_cleavage_aa():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df["enzN"] = df.apply(
        lambda x: peptide_forest.prep.test_cleavage_aa(
            x["Sequence Pre AA"], x["Sequence Start"]
        ),
        axis=1,
    )
    assert all(df[df["engine"] == "omssa"]["enzN"]) is True


def test_sequence_aa_c():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df["enzC"] = df.apply(
        lambda x: peptide_forest.prep.test_sequence_aa_c(
            x["Sequence"][-1], x["Sequence Post AA"]
        ),
        axis=1,
    )
    assert all(df[df["Comments"] == "test_sequence_aa_c"]) is True


def test_count_missed_cleavages():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    df["enzInt"] = df["Sequence"].str.count(r"[R|K]")
    assert df[df["Comments"] == "Two missed cleavages"]["enzInt"].unique() == 3
