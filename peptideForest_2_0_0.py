# Peptide Forest 2.0.0
import peptideForest
import json
import multiprocessing
import os
import pprint


def main(
    output_file=None,
    classifier="RF-reg",
    enzyme="trypsin",
    n_train=5,
    n_eval=5,
    q_cut=0.01,
    q_cut_train=0.1,
    train_top_data=True,
    use_cross_validation=True,
    frac_tp=0.9,
    sample_frac=1.0,
    plot_dir="./plots/",
    plot_prefix="Plot",
    initial_engine="msgfplus",
    show_plots="True",
    dpi=300,
):
    """
    Extract features from training set, impute missing values, fit model and make prediction.

    Args:
        output_file (str, optional): path to save new dataframe to, do not save if None (default)
        classifier (str, optional): name of the classifier to use
        enzyme (str, optional): name of the enzyme used during sample preparation
        n_train (int, optional): number of training iterations
        n_eval (int, optional): number of evaluation iterations
        q_cut (float, optional): cut-off for q-values below which a target PSM is counted as top-target
        q_cut_train (float, optional):  cut-off for q-values below which a target PSM is counted as top-target,
                                        if train_top_data=True
        train_top_data (bool, optional): only use top data (0.1% q-value) for training
        use_cross_validation (bool, optional): use of cross-validation
        plot_dir (str, optional): directory to save plots to
        plot_prefix (str,optional): filename prefix for generated plots
        initial_engine (str, optional): name of initial engine
        frac_tp (float, optional): estimate of fraction of true positives in target dataset
        sample_frac (float, optional): ratio of decoy PSMs to target and decoy PSMs
        show_plots (bool, optional): display plots
        dpi (int, optional): plotting resolution
    """

    timer = peptideForest.runtime.PFTimer()
    totaltimer = peptideForest.runtime.PFTimer()
    totaltimer["total_run_time"]

    # Import hyperparameter and path adict from .json file
    with open("config/hyperparameters.json") as jd:
        hyperparameters = json.load(jd)
    jd.close()
    with open("config/ursgal_path_dict.json") as upd:
        path_dict = json.load(upd)
    upd.close()

    # Add core count information
    hyperparameters["RF"]["n_jobs"] = multiprocessing.cpu_count() - 1
    hyperparameters["RF-reg"]["n_jobs"] = multiprocessing.cpu_count() - 1

    # Load hyperparameters for specified classifier
    hyperparameter_dict = hyperparameters[classifier]
    print(f"Peptide Forest initialised with classifier: {classifier}\n")
    print("Using hyperparameters:")
    pprint.pprint(hyperparameter_dict)
    print()

    if enzyme != "trypsin":
        raise ValueError("Enzymes other than trypsine not implemented yet.")

    # Load data and combine in one dataframe
    input_df = peptideForest.setup_dataset.combine_ursgal_csv_files(
        path_dict, output_file
    )

    # Extract features from dataframe
    print("\nExtracting features...")
    timer["features"]
    df_training, old_cols, feature_cols = peptideForest.setup_dataset.extract_features(
        input_df
    )
    n_features = str(len(feature_cols))
    print(f"Extracted {n_features} features in", "{features}".format(**timer))

    n_rows_df = input_df.shape[0]
    if n_rows_df < 100:
        raise Exception(
            f"Too few idents to run machine learning. DataFrame has only {n_rows_df} rows"
        )

    # [TRISTAN] evtl. nicht truncaten sondern mit versionsnummer übernehmen aber alles andere entsprechened fixen
    all_engines = list(input_df["engine"].unique())
    all_engines_truncated = []
    for e in all_engines:
        x = e.split("_")
        all_engines_truncated.append(x[0])

    print(
        "Working on results from engines {0} and {1}".format(
            ", ".join(all_engines[:-1]), all_engines[-1]
        )
    )

    # Export dataframe
    if output_file is not None:
        timer["export"]
        df_training.to_csv(output_file.split(".csv")[0] + "-features.csv")
        input_df.to_csv(output_file)
        print("Exported dataframe in {export}".format(**timer))

    # Fit model
    timer["fit_model"]
    df_training["Is decoy"] = df_training["Is decoy"].astype(bool)
    (
        clfs,
        psms,
        psms_avg,
        psms_engine,
        df_training,
        df_feature_importance,
    ) = peptideForest.models.fit(
        df_training=df_training,
        classifier=classifier,
        n_train=n_train,
        n_eval=n_eval,
        train_top_data=train_top_data,
        use_cross_validation=use_cross_validation,
        feature_cols=feature_cols,
        initial_score_col="Score_processed_{0}".format(initial_engine),
        hyperparameters=hyperparameter_dict,
        q_cut=q_cut,
        q_cut_train=q_cut_train,
        frac_tp=frac_tp,
        sample_frac=sample_frac,
    )

    print("Fitted model in {fit_model}".format(**timer))
    print("\nFeature importance:")
    print("Score_processed_{0}".format(initial_engine))
    print(df_feature_importance.head(), "\n")

    timer["analysis"]
    # Analyse results
    df_training = peptideForest.results.analyse(
        df_training,
        initial_engine,
        q_cut,
        frac_tp=frac_tp,
        top_psm_only=True,
        all_engines_truncated=all_engines_truncated,
        plot_prefix=plot_prefix,
        plot_dir=plot_dir,
        classifier=classifier,
    )

    # Plot results:
    if os.path.exists(plot_dir) is False:
        os.mkdir(plot_dir)

    peptideForest.plot.all(
        df_training,
        classifier=classifier,
        all_engines_truncated=all_engines_truncated,
        methods=None,
        plot_prefix=plot_prefix,
        plot_dir=plot_dir,
        show_plot=show_plots,
        dpi=dpi,
        initial_engine=initial_engine,
    )

    # # Get shifted PSMs [TRISTAN] Soll ich das wieder zurück in basic schieben?
    # peptideForest.results.get_deltas(
    #     df_training,
    #     all_engines_truncated=all_engines_truncated,
    #     plot_prefix=plot_prefix,
    #     plot_dir=plot_dir,
    #     classifier=classifier,
    # )

    print("\nFinished analysing results and plotting in {analysis}".format(**timer))
    if output_file:
        timer["writing_output"]
        df_training.to_csv(output_file, index=False)
        print(
            "\nWrote output csv to", output_file, "in {writing_output}".format(**timer)
        )

    print("\nComplete run time: {total_run_time}".format(**totaltimer))


if __name__ == "__main__":
    main()
