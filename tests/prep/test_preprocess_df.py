import pandas as pd
import pytest
pd.set_option("max_columns", 100)
import peptideForest

path_dict_small = {
    "tests/_data/mascot_dat2csv_1_0_0.csv": {
        "engine": "mascot",
        "score_col": "Mascot:Score"
    }
}

def test_preprocessing_drops_sequences_with_X():
    df_in = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_small)
    
    df_out = peptideForest.prep.preprocess_df(df_in)
    
    df_out['Number of X'] = df_out['Sequence'].str.count('X')
    assert df_out['Number of X'].unique() == 0

def test_preprocessing_drops_sequences_that_are_decoys_and_targets():
    df_in = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict_small)
    df_out = peptideForest.prep.preprocess_df(df_in)
    assert df_out['test_field'].unique() == 'good!'



