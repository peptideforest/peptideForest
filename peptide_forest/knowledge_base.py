parameters = {
    "remove_cols": [
        "Accuracy (ppm)",
        "Retention Time (s)",
        "Complies search criteria",
        "Conflicting uparam",
        # "Spectrum Title",
        "Mass Difference",
        "Raw data location",
        "Rank",
        "Calc m/z",
    ],
    "proton": 1.00727646677,
    "remove_after_row_features": [
        "Sequence Post AA",
        "Sequence Pre AA",
        "Sequence Start",
        "Sequence Stop",
        "_score_min",
        "_score_max",
        "Score",
        "Exp m/z",
        "uCalc m/z",
    ],
    "non_trainable_columns": {
        "Spectrum Title",
        "Spectrum ID",
        "Sequence",
        "Modifications",
        "Is decoy",
        "Protein ID",
        "Search Engine",
        "uCalc Mass",
        "model_score",
        "model_score_all",
        "model_score_train",
        "model_score_train_all",
        "prev_score_train",
    },
    "hyperparameters": {"n_estimators": 100, "max_depth": 22, "max_features": 7},
}
