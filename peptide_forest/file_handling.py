"""Tools for loading and saving data."""
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

import peptide_forest.knowledge_base
import peptide_forest.sample


def load_csv_with_sampling_information(
        file, cols: List[str], sample_dict: dict
) -> pd.DataFrame:
    """Load a csv file with sampling information given as either the number of lines to
    sample or a dictionary of lines to keep.

    Args:
        file (str): path to csv file
        cols (list): columns to be loaded
        sample_dict (dict): dictionary of lines to keep

    Returns:
        df (pd.DataFrame): input data
    """
    file_size = sum(1 for l in open(file))

    lines_to_keep = sample_dict.get(file, None)
    if lines_to_keep is None:
        return None
    else:
        skip_idx = list(set(range(1, file_size)) - set(lines_to_keep))

    df = pd.read_csv(file, usecols=cols, skiprows=skip_idx)

    return df


def shared_columns(files: List[str]) -> List[str]:
    """Get list of columns shared across all files.

    Args:
        files (list): list of files to be compared

    Returns:
        shared_cols (list): list of columns shared across all files
    """
    all_cols = []
    for file in files:
        with open(file, encoding="utf-8-sig") as f:
            all_cols.append(set(f.readline().replace("\n", "").split(",")))
    shared_cols = list(
        set.intersection(*all_cols)
        - set(peptide_forest.knowledge_base.parameters["remove_cols"])
    )
    return shared_cols


def drop_duplicates_with_log(df: pd.DataFrame, file: str) -> pd.DataFrame:
    """Try dropping duplicated rows from a dataframe and warn if duplicates were found.

    Args:
        df (pd.DataFrame): dataframe to be checked
        file (str): name of file to be checked

    Returns:
        df (pd.DataFrame): dataframe with duplicated rows dropped
    """
    init_len = len(df)
    df.drop_duplicates(inplace=True)
    rows_dropped = init_len - len(df)
    if rows_dropped != 0:
        logger.warning(f"{rows_dropped} duplicated rows were dropped in {file}.")
    return df


def create_dir_if_not_exists(filepath: str, dir_name: str) -> None:
    """Create directory if it does not exist.

    Args:
        filepath (str): path to file
        dir_name (str): name of directory to be created
    """

    parent_dir = Path(filepath)

    dir_path = parent_dir / dir_name

    if not dir_path.exists():
        dir_path.mkdir()
        logger.info(f"Created directory {dir_path} in output folder.")
    else:
        logger.warning(
            f"Directory {dir_path} already exists. Writing into this "
            f"directory. Note this can lead to the overwriting of files."
        )
