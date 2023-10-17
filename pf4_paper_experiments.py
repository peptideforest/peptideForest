import re
from itertools import permutations

from iterative_training_helpers import (
    create_run_config,
    get_split_options,
)

# define a model to be trained and a criterion for iterations (e.g. split by reps)
# code then handles:
#   - training the base model
#   - repeats until the exhaustion of all possible combinations (prove training order does not matter)
#   - evaluating all splits (seen/unseen) by the trained model
#   - training each possible increment

"""
- get all options for splits (strain, fraction, enzyme, rep from base dir)
- create run config dirs for each specified split
    - create all possible starting options
    - create all possible training paths
- structure runs
- run
    - train
    - cross eval
"""


if __name__ == "__main__":
    data_dir = "./data"
    test_pattern = re.compile(
        r"(?P<fraction>cytosol|membrane)_(?P<group>Big12|FASP_v5)"
    )

    options = get_split_options(data_dir, pattern=test_pattern)

    # create base run configs
    split_by = "strain"
    base_files = []
    for option in options[split_by]:
        file_name, _ = create_run_config(
            data_path=data_dir,
            strain="*" if split_by != "strain" else option,
            fraction="*" if split_by != "fraction" else option,
            enzyme="*" if split_by != "enzyme" else option,
            rep="*" if split_by != "rep" else option,
            filename_pattern=test_pattern,
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
