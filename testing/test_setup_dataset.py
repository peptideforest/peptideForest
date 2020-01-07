import json
import pandas as pd
from peptideForest import setup_dataset
from testing.reference.peptide_forest import setup_dataset as ref_setup_dataset

input_df = pd.DataFrame()


def test_combine_ursgal_csv_files():
    with open("/Users/tr/PycharmProjects/peptideForest/config/ursgal_path_dict.json") as j:
        path_dict = json.load(j)
    j.close()

    reference_combine_ursgal = ref_setup_dataset.combine_ursgal_csv_files(path_dict=path_dict)
    test_combine_ursgal = setup_dataset.combine_ursgal_csv_files(path_dict=path_dict)
    reference_combine_ursgal.to_csv("/Users/tr/PycharmProjects/peptideForest/testing/reference/input_df.csv")
    assert test_combine_ursgal.equals(reference_combine_ursgal)


def test_extract_features():

    input_df = pd.read_csv("/Users/tr/PycharmProjects/peptideForest/testing/reference/input_df.csv")

    # Actual test
    test_df_training, test_old_cols, test_feature_cols = setup_dataset.extract_features(input_df)

    input_df = pd.read_csv("/Users/tr/PycharmProjects/peptideForest/testing/reference/input_df.csv")
    ref_df_training, ref_old_cols, ref_feature_cols = ref_setup_dataset.make_dataset(
            input_df=input_df, combine_engines=True, keep_ursgal=False,
        )
    equal_df_training = test_df_training.equals(ref_df_training)
    equal_old_cols = all(test_old_cols == ref_old_cols)
    equal_feature_cols = test_feature_cols == ref_feature_cols

    ref_df_training.to_csv("/Users/tr/PycharmProjects/peptideForest/testing/reference/df_training.csv")

    assert equal_df_training and equal_old_cols and equal_feature_cols
