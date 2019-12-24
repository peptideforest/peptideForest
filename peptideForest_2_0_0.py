# Peptide Forest 2.0.0
import json
import multiprocessing
import pprint
import peptideForest


def main(
        path_dict="config/path_dict.json",
        output_file=None,
        classifier="RF-reg",
        n_train=5,
        n_eval=5,
        q_cut=0.01,
        q_cut_train=0.1,
        train_top_data=True,
        use_cross_validation=True,
        plot_dir="./plots/",
        plot_prefix="",
        initial_engine="msgfplus",
):
    """
    Extract features from training set, impute missing values, fit model and make prediction.

    Args:
        path_dict (str): path to ursgal path dict as .json
        output_file (str, optional): path to save new data frame to, do not save if None (default)
        classifier (str, optional): name of the classifier
        n_train (int, optional): number of training iterations
        n_eval (int, optional): number of evaluation iterations
        q_cut (float, optional): cut-off for q-values below which a target PSM is counted as top-target
        q_cut_train (float, optional): cut-off for q-values below which a target PSM is counted as top-target, if train_top_data=True
        train_top_data (bool, optional): only use top data (0.1% q-value) for training
        use_cross_validation (bool, optional): use of cross-validation
        plot_dir (str, optional): path to save plots to
        plot_prefix (str,optional): filename prefix for generated plots
        initial_engine (str, optional): initial engine

    """

    timer = peptideForest.runtime.PFTimer()
    timer["total_run_time"]

    # Import hyper parameter and path dict from .json file
    with open("config/hyper_parameters.json") as jd:
        hyper_parameters = json.load(jd)
    jd.close()
    with open("config/ursgal_path_dict.json") as pd:
        path_dict = json.load(pd)
    pd.close()

    # Add core count information
    hyper_parameters["RF"]["n_jobs"] = multiprocessing.cpu_count() - 1
    hyper_parameters["RF-reg"]["n_jobs"] = multiprocessing.cpu_count() - 1

    # Load hyper parameters for specified classifier
    hyper_parameter_dict = hyper_parameters[classifier]
    print(f"Peptide Forest initialised with classifier: {classifier}")
    print("Using hyper parameters:")
    pprint.pprint(hyper_parameter_dict)

    # So soll es am Ende aussehen:
    # input_df = peptideForest.setup_dataset.combine_ursgal_csv_files(path_dict, output_file)
    # df_training, old_data, feature_cols = peptideForest.setup_dataset.extract_features(input_df)
    # df_training.to_csv(output_file.split(".csv")[0] + "-features.csv")
    # input_df.to_csv(output_file)
    # messy code in between
    # (Tuple) = peptideForest.models.fit(df_training, classifier, n_train, n_eval, train_top_data, use_cross_validation, feature_cols, hyper_parameters, q_cut, q_train)
    # half of the set is not used again?
    # New function doing all the analyses couple lines are completely useless in the old version (ll. 268-270)
    # df_training = peptideForest.results.analyse(df_training, initial_engine, q_val_cut)
    # Plot results:
    # peptideForest.plot.all()

    print("Complete run time: {total_run_time} min".format(**timer))


if __name__ == "__main__":
    main(output_file="output.csv")
