import json
import re
from itertools import permutations
from loguru import logger
from pathlib import Path

from iterative_training_helpers import (
    create_run_config,
    get_split_options,
    generate_accepted_groups_dict,
    check_for_trained_models,
    get_model_name_str,
    PATTERN,
)
from run_peptide_forest import run_peptide_forest

URSGAL_OUTPUTS = Path.cwd().parent / "ursgalOutputs"


def run_eval(config_dir, filename):
    with open(config_dir + "/" + filename, "r") as f:
        config_dict = json.load(f)
    config_dict["conf"] = ml_model_config_eval
    config_dict["n_train"] = 0

    cross_eval_prefix = (
        f"data>{filename.split('config_')[1].split('.json')[0]}" f"_crosseval>model|"
    )
    cross_eval_output_file = output_name.replace("results", cross_eval_prefix)
    cross_eval_config_file = cross_eval_prefix + filename
    with open(config_dir + "/" + cross_eval_config_file, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    logger.info(
        f"Evaluating model: {model_name} with config: "
        f"{cross_eval_prefix.replace('_crosseval>model|', '')}"
    )
    try:
        run_peptide_forest(
            config_path=config_dir + "/" + cross_eval_config_file,
            output=output_dir + "/" + cross_eval_output_file,
        )
    except Exception as e:
        logger.error("e")
        logger.warning(f"Could not complete eval of {cross_eval_config_file}.")


if __name__ == "__main__":
    data_dir = (
        URSGAL_OUTPUTS
        / "paperPXD021874"
        / "pyiohat_1_7_1_w1_7ef0257438c92c90c4aebaee159757a3"
    )
    model_dir = "./models"
    output_dir = "./outputs"
    config_dir = "./config_files"
    pattern = PATTERN
    split_by = "strain"

    options = get_split_options(data_dir, pattern=pattern)

    # create base run config_files
    base_files = []
    for accepted_group_values in generate_accepted_groups_dict(split_by, options):
        file_name, _ = create_run_config(
            data_path=data_dir,
            filename_pattern=pattern,
            accepted_re_group_values=accepted_group_values,
            config_dir=config_dir,
        )
        base_files.append(file_name)
    logger.info("Finished creating config directories for iterative train run.")

    # create run tree (base, evals, iterations)
    training_paths = permutations(base_files, len(base_files))

    for n, training_path in enumerate(training_paths):
        logger.info(f"Starting training path {n+1}")
        for i, base_file in enumerate(training_path):
            model_name = get_model_name_str(training_path, i)
            with open(config_dir + "/" + base_file, "r") as f:
                config_dict = json.load(f)

            model_exists = check_for_trained_models(model_name, model_dir, config_dict)
            if model_exists:
                logger.info(
                    f"Model {model_name} has already been trained. Skipping to"
                    f" next training iteration."
                )
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

            logger.info(f"Training Model: {model_name}")
            try:
                run_peptide_forest(
                    config_path=config_dir + "/" + base_file,
                    output=output_dir + "/" + output_name,
                )
            except Exception as e:
                # todo: use more specific exceptions
                logger.error(e)
                logger.warning(
                    f"Training of model {model_name} failed. Skipping to "
                    f"next training path."
                )
                break

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
                run_eval(config_dir=config_dir, filename=file)
