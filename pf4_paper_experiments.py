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
        r".+_(?P<fraction>cytosol|membrane)_.*(?P<group>Big12|FASP_v5)"
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
        for i, base_file in enumerate(training_path):
            model_name = get_model_name_str(training_path, i)
            with open(config_dir + "/" + base_file, "r") as f:
                config_dict = json.load(f)

            model_exists = check_for_trained_models(model_name, model_dir, config_dict)
            if model_exists:
                continue

            model_type = "xgboost"  # xgboost or random_forest
            file_extension = ".json" if model_type == "xgboost" else ".pkl"

            trained_model_name = get_model_name_str(training_path, i - 1)
            trained_model_path = (
                model_dir + "/" + trained_model_name + file_extension
                if trained_model_name is not None
                else None
            )
            # training run with base_file
            ml_model_config_train = {
                "model_type": model_type,
                "trained_model_path": trained_model_path,
                "eval_model_path": None,
                "model_output_path": model_dir + "/" + model_name + file_extension,
            }
            config_dict["conf"] = ml_model_config_train
            with open(config_dir + "/" + base_file, "w") as json_file:
                json.dump(config_dict, json_file, indent=4)

            output_name = model_name.replace("model_", "results_") + ".csv"
            run_peptide_forest(
                config_path=config_dir + "/" + base_file,
                output=output_dir + "/" + output_name,
            )

            # eval runs
            ml_model_config_eval = {
                "model_type": model_type,
                "trained_model_path": None,
                "eval_model_path": model_dir + "/" + model_name + file_extension,
                "model_output_path": None,
            }
            for file in base_files:
                if file is base_file:
                    continue
                with open(config_dir + "/" + file, "r") as f:
                    config_dict = json.load(f)
                config_dict["conf"] = ml_model_config_eval
                config_dict["n_train"] = 0

                cross_eval_prefix = (
                    f"data>{file.split('config_')[1].split('.json')[0]}"
                    f"_crosseval>model|"
                )
                cross_eval_output_file = output_name.replace(
                    "results", cross_eval_prefix
                )

                cross_eval_config_file = cross_eval_prefix + file
                with open(config_dir + "/" + cross_eval_config_file, "w") as json_file:
                    json.dump(config_dict, json_file, indent=4)
                run_peptide_forest(
                    config_path=config_dir + "/" + cross_eval_config_file,
                    output=output_dir + "/" + cross_eval_output_file,
                )
