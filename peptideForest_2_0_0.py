# Peptide Forest 2.0.0
import json
import multiprocessing
import os
import pprint
from collections import defaultdict as ddict

import click
import pandas as pd
from treeinterpreter import treeinterpreter as ti

import peptide_forest_old as peptide_forest


@click.command()
@click.option(
    "--output_file",
    "-o",
    required=False,
    help="Outfile file in csv format for data after training",
)
@click.option(
    "--ursgal_json",
    "-u",
    required=True,
    help="Ursgal path json containing csv paths and columns",
)
@click.option(
    "--calculate_local_importance",
    "-cli",
    is_flag=True,
    default=False,
    help="Calculate local importance for all input observations",
)
@click.option(
    "--initial_engine",
    "-i",
    required=True,
    help="Initial engine to sort results",
)
def main(
    output_file=None,
    ursgal_json=None,
    calculate_local_importance=False,
    initial_engine=None,
):
    run_peptide_forest(
        output_file=output_file,
        ursgal_path_dict_json=ursgal_json,
        calculate_local_importance=calculate_local_importance,
        initial_engine=initial_engine,
    )


def run_peptide_forest(
    output_file=None,
    min_data=0.7,
    classifier="RF-reg",
    enzyme="trypsin",
    n_train=10,
    n_eval=10,
    q_cut=0.01,
    q_cut_train=0.1,
    train_top_data=True,
    use_cross_validation=True,
    frac_tp=0.9,
    sample_frac=1.0,
    plot_dir="./plots/",
    plot_prefix="Plot",
    # initial_engine="msgfplus_v2018_06_28",
    initial_engine="msgfplus_v2019_04_18",
    # show_plots=False,
    # dpi=300,
    ursgal_path_dict_json=None,
    calculate_local_importance=False,
    peptide_forest_parameter_json=None,
):
    """
    Extract features from training set, impute missing values, fit model and make prediction.

    Args:
        output_file (str, optional): path to save new dataframe to, do not save if None (default)
        min_data (float):   minimum fraction of spectra for which we require that there are at least 2 or 3 respectively
                            PSMs to calculate delta scores
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
            (depracted ... all plots are done via jupyter)
    """

    timer = peptide_forest.runtime.PFTimer()
    totaltimer = peptide_forest.runtime.PFTimer()
    totaltimer["total_run_time"]

    if output_file is None:
        output_file = "peptide_forest_output.csv"

    if ursgal_path_dict_json is None:
        ursgal_path_dict_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config",
            "ursgal_path_dict.json",
        )
    with open(ursgal_path_dict_json) as upd:
        path_dict = json.load(upd)

    # Import hyperparameter and path adict from .json file
    hyperparameters_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "hyperparameters.json",
    )
    with open(hyperparameters_json) as jd:
        hyperparameters = json.load(jd)
    # Add core count information
    # hyperparameters["RF"]["n_jobs"] = multiprocessing.cpu_count() - 1
    hyperparameters["RF-reg"]["n_jobs"] = multiprocessing.cpu_count() - 1

    # Load hyperparameters for specified classifier
    hyperparameter_dict = hyperparameters[classifier]
    print(f"Peptide Forest initialised with classifier: {classifier}\n")
    print("Using hyperparameters:")
    pprint.pprint(hyperparameter_dict)
    print()

    if enzyme != "trypsin":
        raise ValueError("Enzymes other than trypsin not implemented yet.")

    else:
        cleavage_site = "C"

    # Load data and combine in one dataframe
    input_df = peptide_forest.setup_dataset.combine_ursgal_csv_files(path_dict)

    # Extract features from dataframe
    print("\nExtracting features...")
    timer["features"]
    features = ddict(set)  # new place to collec the features
    features["_id_columns"] = set(
        [
            "Spectrum ID",
            "Spectrum Title",
            "Sequence",
            "Modifications",
            "Protein ID",
            "Is decoy",
            "engine",
        ]
    )

    all_engines_version = list(input_df["engine"].unique())

    print(
        "Working on results from engines {0} and {1}".format(
            ", ".join(all_engines_version[:-1]), all_engines_version[-1]
        )
    )
    n_rows_df = input_df.shape[0]

    if n_rows_df < 100:
        raise Exception(
            f"Too few idents to run machine learning. DataFrame has only {n_rows_df} rows"
        )

    df_training, features = peptide_forest.setup_dataset.extract_features(
        input_df,
        cleavage_site=cleavage_site,
        min_data=min_data,
        path_dict=path_dict,
        features=features,
    )

    print("Final feature columns:", features["final_features"])
    # Fit model
    print(f"Extracted all features in", "{features}".format(**timer))
    print("\nFitting Model ...")
    timer["fit_model"]

    (
        clfs,
        psms,
        psms_avg,
        psms_engine,
        df_training,
        df_feature_importance,
    ) = peptide_forest.models.fit(
        df_training=df_training,
        classifier=classifier,
        n_train=n_train,
        n_eval=n_eval,
        train_top_data=train_top_data,
        use_cross_validation=use_cross_validation,
        feature_cols=features["final_features"],
        initial_score_col="Score_processed_{0}".format(initial_engine),
        hyperparameters=hyperparameter_dict,
        q_cut=q_cut,
        q_cut_train=q_cut_train,
        frac_tp=frac_tp,
        sample_frac=sample_frac,
    )

    print("Fitted model in {fit_model}".format(**timer))
    print("\nFeature importance:")
    print("Initial engine: Score_processed_{0}".format(initial_engine))
    print(df_feature_importance.head(15), "\n")

    # Plot results:
    if os.path.exists(plot_dir) is False:
        os.mkdir(plot_dir)

    timer["analysis"]
    # Analyse results
    df_training = peptide_forest.results.analyse(
        df_training,
        initial_engine,
        q_cut,
        frac_tp=1.0,  # for true fdr calculation frac_tp is set to 1.0
        top_psm_only=True,
        all_engines_version=all_engines_version,
        plot_prefix=plot_prefix,
        plot_dir=plot_dir,
        classifier=classifier,
    )

    # # Generate local importance .csv
    if calculate_local_importance is True:
        print("Calculating local importances...")
        prediction, bias, contributions = ti.predict(
            clfs[0][0][0], df_training[features["final_features"]]
        )
        local_importance = pd.DataFrame(
            data=abs(contributions), columns=features["final_features"]
        )
        local_importance = local_importance.div(local_importance.sum(axis=1), axis=0)
        local_importance = local_importance[
            local_importance.columns[local_importance.median(axis=0) > 0.01]
        ]
        local_importance.to_csv(f"{output_file}_local_importance.csv")

    # peptide_forest.plot.all(
    #     df_training,
    #     classifier=classifier,
    #     all_engines_version=all_engines_version,
    #     methods=None,
    #     plot_prefix=plot_prefix,
    #     plot_dir=plot_dir,
    #     show_plot=show_plots,
    #     dpi=dpi,
    #     initial_engine=initial_engine,
    # )

    # print(
    #     "\nFinished analysing results and plotting in {analysis}".format(
    #         **timer
    #     )
    # )

    timer["writing_output"]
    df_training.to_csv(output_file, index=False)
    print(
        "\nWrote output csv to",
        output_file,
        "in {writing_output}".format(**timer),
    )

    print("\nComplete run time: {total_run_time}".format(**totaltimer))


if __name__ == "__main__":
    main()
