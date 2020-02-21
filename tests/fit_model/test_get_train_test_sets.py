import pandas as pd

import peptideForest


def test_get_train_test_sets():

    df = pd.DataFrame(
        {
            "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "Is decoy": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    df["Is decoy"] = df["Is decoy"].astype(bool)
    comments = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    df["Comments"] = comments

    # With cross validation
    train_set = peptideForest.models.get_train_test_sets(df, use_cross_validation=True)
    assert len(train_set) == 3
    assert set(comments) == set(list(pd.concat(train_set)["Comments"]))

    # Without cross validation
    train_set = peptideForest.models.get_train_test_sets(df, use_cross_validation=False)
    assert len(train_set) == 2
    assert len(train_set[0]) == len(train_set[1]) == 6+3
    targets = ["a", "b", "c", "d", "e", "g"]
    train_set = pd.concat(train_set)
    assert sorted(list(train_set["Comments"])) == sorted(comments + targets)
