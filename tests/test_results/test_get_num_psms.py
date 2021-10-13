import pandas as pd

import peptide_forest

df_method = pd.DataFrame(
    {
        "top_target_ref": [0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        "top_target_a": [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        "top_target_b": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        "top_target_c": [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        "top_target_d": [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    }
)

df_q_cut = pd.DataFrame(
    {
        "top_target_a": [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        "top_target_b": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        "q-value_a": [
            0.02856,
            0.00505,
            0.0376,
            0.03653,
            0.01596,
            0.01358,
            0.04238,
            0.01241,
            0.03638,
            0.01777,
        ],
        "q-value_b": [
            0.01281,
            0.01737,
            0.01305,
            0.00071,
            0.00333,
            0.01043,
            0.01642,
            0.00936,
            0.00478,
            0.01279,
        ],
        "Spectrum ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        "Is decoy": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    }
)

solution_by_method = pd.DataFrame(
    {
        "method": [
            "ref",
            "a",
            "b",
            "c",
            "d",
            "any-engine",
            "all-engines",
            "majority-engines",
        ],
        "n_psms": [7, 6, 2, 6, 5, 10, 0, 3],
    }
)


def test_get_num_psms_by_method():
    df_test = peptide_forest.results.get_num_psms_by_method(df=df_method, methods=None)
    df_test = df_test.sort_index()
    assert df_test.equals(solution_by_method)


def test_get_num_psms_by_q_cut():
    df_test = peptide_forest.results.get_num_psms_against_q_cut(
        df=df_q_cut,
        methods=None,
        q_val_cut=None,
        initial_engine="a",
        all_engines_version=["a", "b"],
    )
    solution_by_q_cut = pd.read_csv(
        "tests/_data/solution_get_num_psms_by_q_cut.csv", index_col=0
    )
    df_test = df_test.sort_index(ascending=False).reset_index(drop=True)
    assert all(df_test == solution_by_q_cut)
