"""Knowledge database for peptide forest internal use."""

parameters = {
    "remove_cols": [
        "rank",
        "calc_mz",
        "chemical_composition",
        "spectrum_title",
    ],
    "proton": 1.00727646677,
    "remove_after_row_features": [
        "sequence_post_aa",
        "sequence_pre_aa",
        "sequence_start",
        "sequence_stop",
        "_score_min",
        "_score_max",
        "score",
        "exp_mz",
        "ucalc_mz",
    ],
    "non_trainable_columns": {
        "raw_data_location",
        "retention_time_seconds",
        "spectrum_id",
        "sequence",
        "modifications",
        "is_decoy",
        "protein_id",
        "model_score",
        "model_score_all",
        "model_score_train",
        "model_score_train_all",
        "prev_score_train",
        "reported_by_",
    },
    "hyperparameters_xgboost": {
        "reg_alpha": 50,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    "hyperparameters_random_forest": {
        "n_estimators": 100,
        "max_depth": 22,
        "max_features": 7,
    },
    "conf": {
        "model_type": "random_forest",
        "model_output_path": "peptide_forest_model.pkl",
    },
}
