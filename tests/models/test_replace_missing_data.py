import pandas as pd
import math
import peptide_forest


train_a = pd.DataFrame(
    {
        "colA_delta_score": [2, 4, None, 4, 4],
        "colB_delta_score": [1, 3, None, 3, 3],
        "colC": [0.5, 0.5, None, 0.5, 0.5],
    }
)
train_b = pd.DataFrame(
    {
        "colA_delta_score": [3, 4, 4, 4, 4],
        "colB_delta_score": [2, 3, None, None, 3],
        "colC": [0.5, 0.5, 0.5, 0.5, 0.5],
    }
)
test = pd.DataFrame(
    {
        "colA_delta_score": [2, 4, None, 4, 4],
        "colB_delta_score": [1, 3, None, 3, 3],
        "colC": [0.5, 0.5, None, 0.5, 0.5],
    }
)
training_data = [train_a, test]


def test_replace_missing_data_cv():
    a, b, t = peptide_forest.models.replace_missing_data_cv(train_a, train_b, test)
    # Check a
    assert list(a.iloc[1, :]) == [4, 3, 0.5]
    assert list(a.iloc[2, :-1]) == [2, 1] and math.isnan(a.iloc[2][2])
    # Check b
    assert list(b.iloc[1, :]) == [4, 3, 0.5]
    assert list(b.iloc[2, :]) == [4, 1, 0.5]
    # Check t
    assert list(t.iloc[1, :]) == [4, 3, 0.5]
    assert list(t.iloc[2, :-1]) == [2, 1] and math.isnan(t.iloc[2][2])


def test_replace_missing_data_top_targets():
    test = peptide_forest.models.replace_missing_data_top_targets(
        training_data=training_data
    )
    a = test[0]
    t = test[1]
    # Check a
    assert list(a.iloc[1, :]) == [4, 3, 0.5]
    assert list(a.iloc[2, :-1]) == [2, 1] and math.isnan(a.iloc[2][2])
    # Check t
    assert list(t.iloc[1, :]) == [4, 3, 0.5]
    assert list(t.iloc[2, :-1]) == [2, 1] and math.isnan(t.iloc[2][2])
