import glob
import json
import os.path
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCORE_COL_MAPPING = {
    "omssa_2_1_9": "omssa:evalue",
    "msgfplus_2021_03_22": "ms-gf:evalue",
    "xtandem_vengeance": "x!tandem:hyperscore",
    "msfragger_3_0": "msfragger:hyperscore",
}

PATTERN = re.compile(
    r".*_(?P<strain>[^_]+)_(?P<fraction>[^_]+)_(?P<enzyme>[^_]+)_(?P<rep>[^_]+)\.raw$"
)


def extract_variables(filename, pattern):
    match = pattern.match(filename)
    if match:
        return match.groupdict()
    return None


def get_split_options(data_path, pattern, col="raw_data_location"):
    """

    Args:
        data_path: str to directory with csvs
        pattern: regex pattern containing groups to be matched
        col: str column name used for extracting strings

    Returns: defaultdict mapping group names to all possible values for matches found in
    the directory

    """
    split_options = defaultdict(set)
    csv_files = glob.glob(f"{data_path}/*.csv")
    for file in csv_files:
        df = pd.read_csv(file)
        raw_file = df[col].unique()[0]
        check_uniform_column_content(df, col)
        vars_file = extract_variables(raw_file, pattern=pattern)
        for variable, option in vars_file.items():
            split_options[variable].add(option)

    return split_options


def create_run_config(
    data_path,
    filename_pattern,
    accepted_re_group_values,
    initial_engine="omssa_2_1_9",
    write_file=True,
    config_dir="./",
):
    csv_files = glob.glob(f"{data_path}/*.csv")
    config = dict()

    for file in csv_files:
        df = pd.read_csv(file)
        engine = df["search_engine"].unique()[0]
        raw_file = df["raw_data_location"].unique()[0]
        check_uniform_column_content(df, "search_engine")
        if not is_matching_filename(
            filename=Path(raw_file).stem,
            pattern=filename_pattern,
            allowed_values=accepted_re_group_values,
        ):
            continue
        score_col = SCORE_COL_MAPPING[engine]
        config[file] = {"engine": engine, "score_col": score_col}

    config_dict = {"input_files": config, "initial_engine": initial_engine}

    accepted_re_group_values_str = ""
    for group, options in accepted_re_group_values.items():
        accepted_re_group_values_str += f"_{group}|{''.join(options)}"
    filename = f"config_{accepted_re_group_values_str}.json"

    if dataset_name is not None:
        filename = filename.replace("config_", f"config__ds_{dataset_name}__")

    if write_file:
        with open(config_dir + "/" + filename, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

    return filename, config_dict


def is_matching_filename(filename, pattern, allowed_values):
    """Checks the group values of a filename matching a regex pattern are allowed values

    Args:
        filename: str filename
        pattern: r-string specifying groups to be identified
        allowed_values: dict specify the allowed values in the format
            group_name="allowed_value"

        Allowed values are either strings or lists with several allowed strings, * is a
        wildcard option.

        ! regex group names (<group_name>) must match kwargs keys.

    Returns: bool depending on whether the criteria set in the kwargs are met.
    """

    def check_value(val, acceptable_values):
        if acceptable_values == "*":
            return True
        if isinstance(acceptable_values, list) and val in acceptable_values:
            return True
        if val == acceptable_values:
            return True
        return False

    vars_filename = extract_variables(filename, pattern=pattern)

    if allowed_values.keys() != vars_filename.keys():
        raise ValueError("Groups in pattern and given kwargs do not match.")

    for name_element, allowed_values in allowed_values.items():
        if not check_value(vars_filename[name_element], allowed_values):
            return False

    return True


def check_uniform_column_content(df, col):
    if len(df[col].unique()) > 1:
        raise ValueError(f"All values in {col} must be the same.")


def generate_accepted_groups_dict(splitting_group, available_options):
    for option in available_options[splitting_group]:
        yield {
            group: (option if group == splitting_group else "*")
            for group in available_options.keys()
        }


def check_for_trained_models(model_name, model_dir, config_dict):
    """Checks if there are already models with the same name stored in the model_dir.

    Args:
        model_name: str, filename to be checked
        model_dir: str, location of all stored models
        config_dict: config dict of the run

    Returns:
        Bool: true if the model could be found in the directory, false otherwise
    """
    if config_dict is None:
        config_dict = {}

    model_conf = config_dict.get("conf", "*")
    if model_conf != "*":
        model_type = model_conf.get("model_type", "any")
    else:
        model_type = "any"

    if model_type == "random_forest":
        return os.path.isfile(Path(model_dir) / f"{model_name}.pkl")
    elif model_type == "xgboost":
        return os.path.isfile(Path(model_dir) / f"{model_name}.json")
    elif model_type == "any":
        return os.path.isfile(Path(model_dir) / f"{model_name}.pkl") or os.path.isfile(
            Path(model_dir) / f"{model_name}.json"
        )
    else:
        raise ValueError("Unknown model_type")


def get_model_name_str(training_schedule, current_idx):
    """Based on a list of config file names and an index of the list generates a name
    for the model at that position.

    Args:
        training_schedule: list or tuple of several config file names to be trained
            sequentially
        current_idx: index of the config file in the list to generate the model name for

    Returns:
        model_name as str

    """
    if current_idx == -1:
        return None
    model_name = "model_" + ">".join(
        file.split("config_")[1].split(".json")[0]
        for file in training_schedule[: current_idx + 1]
    )
    return model_name
