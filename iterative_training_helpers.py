import glob
import json
from pathlib import Path
import re

import pandas as pd

SCORE_COL_MAPPING = {
    "omssa_2_1_9": "omssa:evalue",
    "msgfplus_2021_03_22": "ms-gf:evalue",
    "xtandem_vengeance": "x!tandem:hyperscore",
    "msfragger_3_0": "msfragger:hyperscore",
}
# DATA_PATH = "./data/newDecoys"

PATTERN = re.compile(
    r".*_(?P<strain>[^_]+)_(?P<fraction>[^_]+)_(?P<enzyme>[^_]+)_(?P<rep>[^_]+)\.raw$"
)


def extract_variables(filename, pattern=PATTERN):
    match = pattern.match(filename)
    if match:
        return match.groupdict()
    return None


def create_run_config(
    data_path,
    strain,
    fraction,
    enzyme,
    rep,
    filename_pattern=PATTERN,
    initial_engine="omssa_2_1_9",
    name="pf4",
    write_file=True,
):
    csv_files = glob.glob(f"{data_path}/*.csv")
    config = dict()

    for file in csv_files:
        df = pd.read_csv(
            file,
        )
        engine = df["search_engine"].unique()[0]
        raw_file = df["raw_data_location"].unique()[0]
        if not is_matching_filename(
            filename=Path(raw_file).stem,
            strain=strain,
            fraction=fraction,
            enzyme=enzyme,
            rep=rep,
            filename_pattern=filename_pattern,
        ):
            continue
        if len(df["search_engine"].unique()) > 1:
            raise ValueError("Too many engines used in one file")
        score_col = SCORE_COL_MAPPING[engine]
        config[file] = {"engine": engine, "score_col": score_col}

    config_dict = {"input_files": config, "initial_engine": initial_engine}

    filename = f"config_{name}_str{''.join(strain)}_frc{''.join(fraction)}_enz{''.join(enzyme)}_rep{''.join(rep)}.json"

    if write_file:
        with open(filename, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

    return filename, config_dict


def is_matching_filename(
    filename, strain, fraction, enzyme, rep, filename_pattern=PATTERN
):
    def check_value(val, acceptable_values):
        if acceptable_values == "*":
            return True
        if isinstance(acceptable_values, list) and val in acceptable_values:
            return True
        if val == acceptable_values:
            return True
        return False

    vars_filename = extract_variables(filename, pattern=filename_pattern)
    strain_val = vars_filename["strain"]
    fraction_val = vars_filename["fraction"]
    enzyme_val = vars_filename["enzyme"]
    rep_val = vars_filename["rep"]

    if not check_value(strain_val, strain):
        return False
    if not check_value(fraction_val, fraction):
        return False
    if not check_value(enzyme_val, enzyme):
        return False
    if not check_value(rep_val, rep):
        return False

    return True
