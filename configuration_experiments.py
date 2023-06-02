import multiprocessing as mp
from uuid import uuid4

import peptide_forest
from peptide_forest.pf_config import PFConfig
from peptide_forest.visualization import plot_q_value_curve, plot_psms_at_qval_threshold


CONFIGS = [
    {
        "name": "base case",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 1},
            "n_folds": {"value": 3},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "5 iterations, n_estimators +=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 5},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, reg_lambda 10000 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_lambda": {"value": 10000, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, learning_rate div=2",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "learning_rate": {"strategy": "/=2"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, reg_alpha 10 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 1},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, max_depth 3+=3",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "max_depth": {"value": 3, "strategy": "+=3"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, q_cut 0.001",
        "config": {
            "q_cut": {"value": 0.001},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, q_cut 0.01 div=10",
        "config": {
            "q_cut": {"value": 0.01, "strategy": "/=10"},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, q_cut 0.0001*=10",
        "config": {
            "q_cut": {"value": 0.0001, "strategy": "*=10"},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "n_jobs": {"value": 1},
        },
    },
]

GRID_CONFIG = [
    {
        "name": "large grid search, 10 iterations, 2 folds, q_cut 0.001 div=2, lambda 10000 div=10",
        "config": {
            "q_cut": {"value": 0.001, "strategy": "/=2"},
            "n_train": {"value": 10},
            "n_spectra": {},
            "n_folds": {"value": 2},
            "n_estimators": {
                "value": 100,
                "grid": [100],
                "strategy": "[+=10, +=50]",
            },
            "max_depth": {"value": 3, "strategy": "[+=3, +=5, _]", "grid": [3]},
            "learning_rate": {
                "value": 0.1,
                "grid": [0.1],
                "strategy": "[/=2, _]",
            },
            "reg_lambda": {"grid": [10000], "strategy": "/=10"},
            "min_split_loss": {},
            "engine_rescore": {},
            "n_jobs": {"value": 7},
        },
    },
# {
#         "name": "large grid search, 10 iterations, 2 folds, q_cut 0.01",
#         "config": {
#             "q_cut": {"value": 0.01},
#             "n_train": {"value": 10},
#             "n_spectra": {},
#             "n_folds": {"value": 2},
#             "n_estimators": {
#                 "value": 100,
#                 "strategy": "[+=5, +=10, +=50]",
#             },
#             "max_depth": {"value": 3, "strategy": "[+=3, +=5, _, -=2]"},
#             "learning_rate": {
#                 "value": 0.1,
#                 "strategy": "[/=10, _, *=10]",
#             },
#             "reg_alpha": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "reg_lambda": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "min_split_loss": {},
#             "engine_rescore": {},
#             "n_jobs": {"value": 1},
#         },
#     },
# {
#         "name": "large grid search, 10 iterations, 2 folds, q_cut 0.0001 *=2",
#         "config": {
#             "q_cut": {"value": 0.0001, "strategy": "*=2"},
#             "n_train": {"value": 10},
#             "n_spectra": {},
#             "n_folds": {"value": 2},
#             "n_estimators": {
#                 "value": 100,
#                 "strategy": "[+=5, +=10, +=50]",
#             },
#             "max_depth": {"value": 3, "strategy": "[+=3, +=5, _, -=2]"},
#             "learning_rate": {
#                 "value": 0.1,
#                 "strategy": "[/=10, _, *=10]",
#             },
#             "reg_alpha": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "reg_lambda": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "min_split_loss": {},
#             "engine_rescore": {},
#             "n_jobs": {"value": 1},
#         },
#     },
# {
#         "name": "large grid search, 10 iterations, 2 folds, q_cut 0.01 /=2",
#         "config": {
#             "q_cut": {"value": 0.01, "strategy": "/=2"},
#             "n_train": {"value": 10},
#             "n_spectra": {},
#             "n_folds": {"value": 2},
#             "n_estimators": {
#                 "value": 100,
#                 "strategy": "[+=5, +=10, +=50]",
#             },
#             "max_depth": {"value": 3, "strategy": "[+=3, +=5, _, -=2]"},
#             "learning_rate": {
#                 "value": 0.1,
#                 "strategy": "[/=10, _, *=10]",
#             },
#             "reg_alpha": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "reg_lambda": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "min_split_loss": {},
#             "engine_rescore": {},
#             "n_jobs": {"value": 1},
#         },
#     },
# {
#         "name": "large grid search, 10 iterations, 2 folds, q_cut 0.0001",
#         "config": {
#             "q_cut": {"value": 0.0001},
#             "n_train": {"value": 10},
#             "n_spectra": {},
#             "n_folds": {"value": 2},
#             "n_estimators": {
#                 "value": 100,
#                 "strategy": "[+=5, +=10, +=50]",
#             },
#             "max_depth": {"value": 3, "strategy": "[+=3, +=5, _, -=2]"},
#             "learning_rate": {
#                 "value": 0.1,
#                 "strategy": "[/=10, _, *=10]",
#             },
#             "reg_alpha": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "reg_lambda": {"grid": [1, 100, 10000], "strategy": "[/=10, _, *=10]"},
#             "min_split_loss": {},
#             "engine_rescore": {},
#             "n_jobs": {"value": 1},
#         },
#     },
]

DEPTH_REG_CONFIG = [
    {
        "name": "1 iterations, n_estimators +=10, reg_alpha 10 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 1},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, reg_alpha 10 div=10, q_cut 0.001",
        "config": {
            "q_cut": {"value": 0.001},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, reg_alpha 10 div=10, q_cut 0.01 div=10",
        "config": {
            "q_cut": {"value": 0.01, "strategy": "/=10"},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "3 iterations, n_estimators +=10, reg_alpha 1000 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 3},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 1000, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "5 iterations, n_estimators +=10, reg_alpha 10 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 5},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "5 iterations, n_estimators +=10, reg_alpha 1000 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 5},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 1000, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "10 iterations, n_estimators +=10, reg_alpha 10 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 10},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "10 iterations, n_estimators +=10, reg_alpha 1000 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 10},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 1000, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
    {
        "name": "10 iterations, n_estimators +=10, reg_alpha 10000 div=10",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 10},
            "n_folds": {"value": 3},
            "n_estimators": {"strategy": "+=10"},
            "reg_alpha": {"value": 10000, "strategy": "/=10"},
            "n_jobs": {"value": 1},
        },
    },
]

DEPTH_N_SPECTRA = [
    {
        "name": "5 iterations, n_estimators +=10, 3000 spectra, q-cut 0.01",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 5},
            "n_spectra": {"value": 3000},
            "n_folds": {"value": 3},
            "n_estimators": {"value": 100, "strategy": "+=10"},
            "max_depth": {"value": 3},
            "n_jobs": {"value": 7},
        },
    },
{
        "name": "5 iterations, n_estimators +=10, 3000 spectra, q-cut 0.001",
        "config": {
            "q_cut": {"value": 0.001},
            "n_train": {"value": 5},
            "n_spectra": {"value": 3000},
            "n_folds": {"value": 3},
            "n_estimators": {"value": 100, "strategy": "+=10"},
            "max_depth": {"value": 3},
            "n_jobs": {"value": 7},
        },
    },
{
        "name": "5 iterations, n_estimators +=10, 3000 spectra, q-cut 0.001, max depth 6",
        "config": {
            "q_cut": {"value": 0.001},
            "n_train": {"value": 5},
            "n_spectra": {"value": 3000},
            "n_folds": {"value": 3},
            "n_estimators": {"value": 100, "strategy": "+=10"},
            "max_depth": {"value": 6},
            "n_jobs": {"value": 7},
        },
    },
{
        "name": "5 iterations, n_estimators +=10, 3000 spectra, q-cut 0.001, max depth 6",
        "config": {
            "q_cut": {"value": 0.001},
            "n_train": {"value": 5},
            "n_folds": {"value": 3},
            "n_spectra": {"value": 3000},
            "n_estimators": {"value": 100, "strategy": "+=10"},
            "max_depth": {"value": 6},
            "n_jobs": {"value": 7},
        },
    },
{
        "name": "5 iterations, n_estimators 200+=10, 3000 spectra, q-cut 0.01, max depth 3",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 5},
            "n_spectra": {"value": 3000},
            "n_folds": {"value": 3},
            "n_estimators": {"value": 200, "strategy": "+=10"},
            "max_depth": {"value": 3},
            "n_jobs": {"value": 7},
        },
    },
]


"""
{
        "name": "base case",
        "config": {
            "q_cut": {"value": 0.01},
            "n_train": {"value": 1},
            "n_spectra": {},
            "n_folds": {"value": 3},
            "n_estimators": {},
            "max_depth": {},
            "learning_rate": {},
            "reg_alpha": {},
            "reg_lambda": {},
            "min_split_loss": {},
            "engine_rescore": {},
            "n_jobs": {"value": 1},
        },
    },
"""

# todo: try with grid searching
# todo: try engine rescore
# todo: try low estimator number
# todo: compare number of folds
# todo: try different estimator strategies (grid search?)

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    for conf in DEPTH_N_SPECTRA:
        output = f"./docker_test_data/{uuid4()}_output.csv"
        pf = peptide_forest.PeptideForest(
            config_path="./docker_test_data/config_local.json",  # args.c
            output=output,  # args.o,
            memory_limit=None,  # args.m,
            max_mp_count=1,  # args.mp_limit,
        )
        pf.config = PFConfig(conf["config"])
        pf.initial_config = pf.config.copy()
        pf.boost()
        # with open("./configs/" + conf["name"], "w") as f:
        #     repr(pf.config)
        files = {conf["name"]: output}  # output
        title = "deep_spec_count__" + conf["name"]
        plot_q_value_curve(files, title=title)
        plot_psms_at_qval_threshold(files, title=title)
