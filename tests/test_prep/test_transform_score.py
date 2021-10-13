import numpy as np

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


def test_transform_score():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_medium)
    stats = peptide_forest.prep.get_stats(df)

    df["Score_processed"] = df.apply(
        lambda row: peptide_forest.prep.transform_score(
            row["Score"], stats[row["engine"]]
        ),
        axis=1,
    )
    assert df["Score_processed"].all() > 0
    assert df[df["engine"] == "mascot"]["Score_processed"].equals(
        df[df["engine"] == "mascot"]["Score"]
    )

    assert df[(df["engine"] == "omssa")]["Score"].min() <= 1e-30
    # ^-- make sure the test data has omssa exception, ie score <=1e-30

    assert ~df[df["engine"] == "omssa"]["Score_processed"].equals(
        -np.log(df[df["engine"] == "omssa"]["Score"])
    )
    # test if conversion is not simply -np.log10 as mascot

    omssa_df = df[(df["engine"] == "omssa") & (df["Score"] >= 1e-30)]
    assert omssa_df["Score_processed"].equals(-np.log10(omssa_df["Score"]))
    # compare if scores above omssa threshold are actualy just -np.log10
