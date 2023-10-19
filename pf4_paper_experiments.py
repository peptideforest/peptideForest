import json
import re
from itertools import permutations

from iterative_training_helpers import (
    create_run_config,
    get_split_options,
    generate_accepted_groups_dict,
    check_for_trained_models,
    get_model_name_str,
)
from run_peptide_forest import run_peptide_forest


if __name__ == "__main__":
    data_dir = "./data"
    model_dir = "./models"
    output_dir = "./outputs"
    config_dir = "./configs"
    test_pattern = re.compile(
        r"(?P<fraction>cytosol|membrane)_(?P<group>Big12|FASP_v5)"
    )
    split_by = "fraction"

    options = get_split_options(data_dir, pattern=test_pattern)

    # create base run configs
    base_files = []
    for accepted_group_values in generate_accepted_groups_dict(split_by, options):
        file_name, _ = create_run_config(
            data_path=data_dir,
            filename_pattern=test_pattern,
            accepted_re_group_values=accepted_group_values,
            config_dir=config_dir,
        )
        base_files.append(file_name)

    # create run tree (base, evals, iterations)
    training_paths = permutations(base_files, len(base_files))

    for training_path in training_paths:
        # todo: build training routine (important: load correct models)
        pass

    ml_model_config = {
        "model_type": "",  # xgboost or random_forest
        "trained_model_path": None,
        "eval_model_path": None,
        "model_output_path": None,
    }

    config = {"conf": ml_model_config}
