"""
Copyright © 2019 by minds.ai, Inc.
All rights reserved

Make a dataset from target-decoy data to be used by percolator iterative method
"""

# pylint: disable=unused-import
import glob
import os
import time
from typing import Any, Dict, Pattern, Set, Tuple, List
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from . import prep as preprocess_features


def combine_ursgal_csv_files(path_dict: Dict, output_file: str = None) -> pd.DataFrame:
  """
  Combine separate ursgal search output csv files and return a dataframe. Also output as new 
  csv file (optional). Takes a dictionary of input csv files, their engines and name of score
  column
  Arguments:
    - path_dict: dictionary containing input csv files and metadata
    - output_file: file to save new dataframe to. If None (defaul), don't save
  Returns:
    - input_df: combined dataframe
  """

  # list of seperate dataframes
  dfs = []

  # columns that are not needed
  cols_to_drop = [
    'Spectrum Title',
    'Mass Difference',
    'Raw data location',
    'Rank', # only XTandem, which doesn have rank .... will rename it in the next release
    'Calc m/z' # too bugy and we use ursgal m/z, i.e. uCalc m/z
  ]

  for file in path_dict.keys():
    start_time = time.time()
    df = pd.read_csv(file)
    df['engine'] = path_dict[file]['engine']
    df['Score'] = df[path_dict[file]['score_col']]
    file_output = file.split('/')[-1]
    total_time = round(time.time() - start_time, 2)
    print(f'Slurping in df for {file_output} in {total_time}s')
    
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    dfs.append(df)
    
  input_df = pd.concat(dfs, sort=True).reset_index(drop=True)

  if output_file:
    input_df.to_csv(output_file)
  
  return input_df
 

def make_dataset(file_name: str = None, input_df: pd.DataFrame = None, engines: List = None,
                 only_top: bool = False,
                 balance_dataset: bool = False,
                 combine_engines: bool = True,
                 keep_ursgal: bool = False) -> Tuple[pd.DataFrame, List, List]:
  """
  Read in a csv file containing raw data from a single experiment, and calculate features if an
  engine name is provided.
  Arguments:
    - file_name: name of csv file, to read data from if input_df = None
    - input_df: input dataframe
    - engines: list of engine names to keep. If None (default), keeps all the engines.
               Engines should be in the lower case form (e.g. mascot, msgfplus, xtandem, omssa).
    - only_top: only keep top targets and decoys, default = False
    - balance_dataset: which targets and decoys to keep.
                       - False (default): keep top target and decoy for each PSM
                       - True: keep top target and decoy for each PSM, only for PSMs where there is
                               one of each
    - combine_engines: combine PSMs for all engines, default is True
    - keep_ursgal: keep original ursgal q-values (if present in dataframe)
  Returns:
    - df: dataframe containing original data and features
    - old_cols: columns initially in the dataframe
    - feature_cols: column names of newly calculated features
  """
  if input_df is None:
    df = pd.read_csv(file_name, index_col=0)
  else:
    df = input_df

  if engines:
    # filter by engine
    df = df.loc[df.engine.apply(lambda x: x.split('_')[0]).isin(engines)]

  # get the features
  df, old_cols, feature_cols = get_features(df, only_top=only_top,
                                            balance_dataset=balance_dataset,
                                            combine_engines=combine_engines,
                                            keep_ursgal=keep_ursgal)

  # replace missing scores with 0
  score_cols = [f for f in df.columns if 'Score_processed' in f]
  df[score_cols] = df[score_cols].fillna(0)


  return df, old_cols, feature_cols


def get_features(df: pd.DataFrame,
                 only_top: bool = False,
                 balance_dataset: bool = False,
                 combine_engines: bool = False,
                 keep_ursgal: bool = False) -> Tuple[pd.DataFrame, List, List]:
  """
  Get features for a dataframe.
  Arguments:
    - df: dataframe containing experiment data
    - only_top: only keep top targets and decoys, default = False
    - balance_dataset: which targets and decoys to keep.
                       - False (default): keep top target and decoy for each PSM
                       - True: keep top target and decoy for each PSM, only for PSMs where there is
                               one of each
    - combine_engines: combine PSMs for all engines
    - keep_ursgal: keep original ursgal q-values (if present in dataframe)
  Returns:
    - df: dataframe containing original data and features
    - old_cols: columns initially in the dataframe
    - feature_cols: column names of newly calculated features
  """
  # get original column names
  old_cols = df.columns
  # calculate the features
  df = preprocess_features.calc_features(df,
                                         only_top=only_top,
                                         balance_dataset=balance_dataset,
                                         combine_engines=combine_engines,
                                         keep_ursgal=keep_ursgal)
  # get a list of the feature names
  feature_cols = list(set(df.columns) - set(old_cols))
  q_value_cols = [f for f in df.columns if 'q-value' in f]
  feature_cols = [f for f in feature_cols if f not in q_value_cols]

  return df, old_cols, feature_cols


def replace_missing_data_cv(train_j: pd.DataFrame, train_k: pd.DataFrame,
                         test: pd.DataFrame) -> pd.DataFrame:
  """
  Replace missing feature values.
  Missing delta_scores: replace with minimum value from training set
  Arguments:
    - train_j, train_k: dataframes containing feature values for training split
    - test: dataframe containing feature values for test split
  Returns:
    - train_j, train_k, test: input dataframes with missing values filled in
  """

  # replace missing delta_scores with minimum value from training data
  del_score_cols = [f for f in train_j.columns if 'delta_score' in f]
  for del_score_col in del_score_cols:
    min_val = min(train_j[del_score_col].min(), train_k[del_score_col].min())
    train_j[del_score_col] = train_j[del_score_col].fillna(min_val)
    train_k[del_score_col] = train_k[del_score_col].fillna(min_val)
    test[del_score_col] = test[del_score_col].fillna(min_val)

  return train_j, train_k, test


def replace_missing_data_top_targets(training_data: List) -> List:
  """
  Replace missing feature values.
  Missing delta_scores: replace with minimum value from training set
  Arguments:
    - training_data: list containing data splits, 0 is training data, 1 is test data
  Returns:
    - training_data: input with missing values filled in
  """

  # replace missing delta_scores with minimum value from training data
  del_score_cols = [f for f in training_data[0].columns if 'delta_score' in f]
  for del_score_col in del_score_cols:
    min_val = training_data[0][del_score_col].min()
    training_data[0][del_score_col] = training_data[0][del_score_col].fillna(min_val)
    training_data[1][del_score_col] = training_data[1][del_score_col].fillna(min_val)

  return training_data


def get_top_target_decoy(df: pd.DataFrame, score_col: str, balanced: bool = False) -> pd.DataFrame:
  """
  Remove all PSMs except the top target and the top decoy for each spectrum from the dataset
  Arguments:
    - df: dataframe containing feature values
    - score_col: column name to rank the PSMs by
    - balanced: return same number of decoys as targets. If there are more targets than decoys,
                decoys are sampled with replacement
  Returns:
    - df: input dataframe with non-top targets/decoys removed
  """
  # get all the top targets
  targets = df[~df['Is decoy']]
  targets = targets.sort_values(score_col, ascending=False).drop_duplicates('Spectrum ID')

  # get all the top decoys
  decoys = df[df['Is decoy']]
  decoys = decoys.sort_values(score_col, ascending=False).drop_duplicates('Spectrum ID')

  # determin sample size
  replace = False
  if balanced:
    n_sample = len(targets)
    if n_sample > len(decoys):
      replace = True
  else:
    n_sample = len(decoys)

  decoys = decoys.sample(n=n_sample, replace=replace)

  # join the data together
  df = pd.concat([targets, decoys])
  df['Is decoy'] = df['Is decoy'].astype(bool)
  return df


def write_percolator_input_files(df: pd.DataFrame, feature_cols: List, file_name: str) -> None:
  """
  Process data into format readable by percolator and write to file
  Arguments:
    - df: dataframe containing experimental data and calculated features
    - feature_cols: names of columns containing features to output
    - file_name: name of file to output to
  """
  df_out = df.loc[:, ['Spectrum ID', 'Is decoy'] + feature_cols + ['Sequence', 'Protein ID']]

  # switch targets to 1, decoys to 0
  tf = {1: -1, 0: 1}
  df_out.loc[:, 'Is decoy'] = df_out['Is decoy'].astype(int).map(tf)

  # change all other True/False to 1/0. First find the boolean columns. Some may be 'strings'
  # rather than boolean, so find them too.
  bool_cols = list(df_out.select_dtypes(include='bool').columns)
  bool_cols = bool_cols + [c for c in feature_cols if 'True' in df_out[c].astype(str).values
                           or 'False' in df_out[c].astype(str).values]
  # set these columns to be boolean, then to be integer columns
  df_out[bool_cols] = df_out[bool_cols].astype(bool).astype(int)

  # rename to percolator column names
  df_out = df_out.rename(columns={'Spectrum ID': 'SpecId',
                                  'Is decoy': 'Label',
                                  'Sequence': 'Peptide',
                                  'Protein ID': 'Proteins'})

  df_top_line = pd.DataFrame(np.zeros([1, df_out.shape[1]]),
                             columns=df_out.columns)
  df_top_line = df_top_line.astype(int)
  df_top_line.loc[:, "SpecId"] = "DefaultDirection"
  df_top_line.loc[:, "Label"] = "-"
  df_top_line.loc[:, "Peptide"] = ""
  df_top_line.loc[:, "Proteins"] = ""

  df_out = pd.concat([df_top_line, df_out], ignore_index=True)

  df_out.to_csv(file_name, sep='\t', index=False)
