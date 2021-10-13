import pandas as pd

pd.set_option("max_columns", 100)
import peptide_forest

path_dict_small_omssa = {
    "tests/_data/omssa_2_1_9.csv": {"engine": "omssa", "score_col": "OMSSA:pvalue"}
}


def test_get_stats_simple():
    df = pd.DataFrame(
        {
            "engine": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "Score": [1, 2, 3, 10, 11, 12, 100, 100],
        }
    )
    stats = peptide_forest.prep.get_stats(df)
    assert stats["A"]["min_score"] == 1
    assert stats["A"]["max_score"] == 3
    assert stats["B"]["min_score"] == 10
    assert stats["B"]["max_score"] == 12
    assert stats["C"]["min_score"] == 100
    assert stats["C"]["max_score"] == 100


def test_get_stats_reset_omssa_crazyness():
    df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict_small_omssa)
    stats = peptide_forest.prep.get_stats(df)
    assert stats["omssa"]["min_score"] == 1e-30
