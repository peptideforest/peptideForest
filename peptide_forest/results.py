from peptide_forest import models, classifier
import numpy as np
import pandas as pd


def mark_top_targets(
    df, q_cut,
):
    """
    Identify which PSMs in a dataframe are top targets (i.e. is the highest ranked target PSM for a given spectrum
    with q-value < i%). Returns results for all Score_processed columns.
    Args:
        df (pd.DataFrame): dataframe with q-values for each PSM
        q_cut (float): q-value to use as a cut off as fraction

    Returns:
        df (pd.DataFrame): dataframe with new columns indicating if the PSM is a top target

    """

    # Drop any top-target columns that are already in the dataframe
    cols = [c for c in df.columns if "top_target" in c]
    df = df.drop(cols, axis=1)

    # Get the column names
    cols_q_val = df.columns[df.columns.str[0:8] == "q-value_"]
    cols_target = ["top_target" + c.split("q-value")[-1] for c in cols_q_val]

    for col_target, col_q_val in zip(cols_target, cols_q_val):
        df_top_targets = df[(df[col_q_val] <= q_cut) & (~df["Is decoy"])]
        df_top_targets = df_top_targets.sort_values(col_q_val)
        df_top_targets = df_top_targets.drop_duplicates("Spectrum ID")
        df[col_target] = False
        df.loc[df_top_targets.index, col_target] = True

    return df


def calc_all_final_q_vals(
    df, frac_tp, top_psm_only, initial_engine,
):
    """
    Calculate q-value for given score column.
    Args:
        df (pd.DataFrame): dataframe with training results
        frac_tp (float): estimate of fraction of true positives in target dataset
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        initial_engine (str):   method which was used to originally rank the PSMs, to be used here as the second ranking
                                column

    Returns:
        df (pd.DataFrame): input dataframe with q-values added as new column

    """
    cols = [c for c in df.columns if "Score_processed_" in c]
    for col in cols:
        if "Score_processed_" in col:
            col = col.split("Score_processed_")[-1]
        score_col = f"Score_processed_{col}"
        q_col = f"q-value_{col}"
        df_scores = models.get_q_vals(
            df,
            score_col,
            frac_tp=frac_tp,
            top_psm_only=top_psm_only,
            initial_engine=initial_engine,
        )
        df[q_col] = 1.0
        df.loc[df_scores.index, q_col] = df_scores["q-value"]
        max_q_val = df.loc[df_scores.index, q_col].max()
        df[q_col] = df[q_col].fillna(max_q_val)
    return df


def get_ranks(df,):
    """
    Add a column with the rank of each PSM for all Score_processed columns.
    Args:
        df (pd.DataFrame): dataframe with scores for each PSM

    Returns:
        df (pd.DataFrame): same dataframe with new columns indicating the ranks

    """
    # Get the column names
    cols_score = df.columns[df.columns.str[0:16] == "Score_processed_"]
    cols_rank = ["rank" + c.split("Score_processed")[-1] for c in cols_score]
    # Get the rank for each score_processed column
    for col_score, col_rank in zip(cols_score, cols_rank):
        df[col_rank] = df[col_score].rank(ascending=False, method="first")

    return df


def get_shifted_psms(df, x_name, y_name, n_return):
    """
    Make dataframes showing which PSMs were top targets before training but no longer are and vise-vera.
    Args:
        df (pd.DataFrame): dataframe with training data and analysed results
        x_name (str): name of method used for baseline (e.g. search engine name)
        y_name (str): name of method used for comparison (e.g. ML model name)

    Returns:
        df_new_top_targets (pd.DataFrame): dataframe containing information on the new top targets
        df_old_top_targets (pd.DataFrame): dataframe containing information on the old top targets
    """
    col_x = f"rank_{x_name}"
    tt_x = f"top_target_{x_name}"

    tt_y = f"top_target_{y_name}"

    # Non top targets that are now top targets
    df_new_top_targets = (
        df[~df[tt_x] & df[tt_y]]
        .sort_values(col_x, ascending=False)
        .copy(deep=True)
    )
    df_new_top_targets = df_new_top_targets.reset_index()
    print(
        f"Number non top targets for {x_name} that are now top targets: {len(df_new_top_targets)}"
    )

    # up_rank_for_spectrum: previously was not top PSM for that spectrum
    df_new_top_targets["up_rank_for_spectrum"] = False
    for i in df_new_top_targets.index:
        spectrum_id = df_new_top_targets.loc[i, "Spectrum ID"]
        sequence = df_new_top_targets.loc[i, "Sequence"]
        protein_id = str(df_new_top_targets.loc[i, "Protein ID"])
        mods = df_new_top_targets.loc[i, "Modifications"]
        df_spectrum = df[df.loc[:, "Spectrum ID"] == spectrum_id]
        if (df_spectrum[f"Score_processed_{x_name}"] != 0).all():
            df_spectrum = df_spectrum.sort_values(
                f"Score_processed_{x_name}", ascending=False
            )
            if any(
                [
                    sequence != df_spectrum["Sequence"].values[0],
                    protein_id
                    != df_spectrum["Protein ID"].astype(str).values[0],
                    mods != df_spectrum["Modifications"].values[0],
                ]
            ):
                df_new_top_targets.loc[i, "up_rank_for_spectrum"] = True

    # Top targets that are now not top targets
    df_old_top_targets = (
        df[df[tt_x] & ~df[tt_y]]
        .sort_values(col_x, ascending=True)
        .copy(deep=True)
    )
    df_old_top_targets = df_old_top_targets.reset_index()
    print(
        f"Number top targets for {x_name} that are not longer top targets: {len(df_old_top_targets)}"
    )
    if n_return is not None:
        df_old_top_targets = df_old_top_targets.head(n_return)

    # down_rank_for_spectrum: moved down the rankings for this spectrum
    # new_best_psm_is_top_target: spectrum has new best match that is also a top target

    df_old_top_targets["down_rank_for_spectrum"] = False
    df_old_top_targets["new_best_psm_is_top_target"] = False

    for i in df_old_top_targets.index:
        spectrum_id = df_old_top_targets.loc[i, "Spectrum ID"]
        sequence = df_old_top_targets.loc[i, "Sequence"]
        protein_id = str(df_old_top_targets.loc[i, "Protein ID"])
        mods = df_old_top_targets.loc[i, "Modifications"]
        df_spectrum = df[df.loc[:, "Spectrum ID"] == spectrum_id]
        if (df_spectrum[f"Score_processed_{x_name}"] != 0).all():
            df_spectrum = df_spectrum.sort_values(
                f"Score_processed_{y_name}", ascending=False
            )
            if any(
                [
                    sequence != df_spectrum["Sequence"].values[0],
                    protein_id
                    != df_spectrum["Protein ID"].astype(str).values[0],
                    mods != df_spectrum["Modifications"].values[0],
                ]
            ):
                df_old_top_targets.loc[i, "down_rank_for_spectrum"] = True
                df_old_top_targets.loc[
                    i, "new_best_psm_is_top_target"
                ] = df_spectrum[f"top_target_{y_name}"].values[0]

    return df_new_top_targets, df_old_top_targets


def get_num_psms_by_method(df, methods):
    """
    Returns a dataframe with the number of PSMs identified as top targets for each method that results are available for
    in input dataframe.
    Args:
        df (pd.DataFrame): dataframe containing results from search engines and ML training
        methods (List): list of methods to use. If None, use all methods.

    Returns:
        df_num_psms: dataframe containing number of PSMs with q_val < q_val_cut for each method
    """
    # Get a list of methods
    # q-values from percolator are ignored, as these are not comparable to our results
    if methods is None:
        methods = [c for c in df.columns if "top_target" in c]
        methods = [c for c in methods if "ursgal" not in c and "from" not in c]

    df_num_psms = df[
        [c for c in df.columns if "top_target_" in c and c in methods]
    ].sum()
    df_num_psms = df_num_psms.to_frame().reset_index()
    df_num_psms.columns = ["method", "n_psms"]
    df_num_psms["method"] = df_num_psms["method"].apply(
        lambda x: x.split("top_target_")[-1]
    )
    engine_cols = [m for m in df.columns if "top_target_" in m]
    engine_cols = [
        m
        for m in engine_cols
        if all(ml_method not in m for ml_method in classifier.ml_methods)
    ]

    if "any" not in df_num_psms["method"].values:
        df_num_psms = df_num_psms.append(
            {
                "method": "any-engine",
                "n_psms": df[engine_cols].any(axis=1).sum(),
            },
            ignore_index=True,
        )
    if "all" not in df_num_psms["method"].values:
        df_num_psms = df_num_psms.append(
            {
                "method": "all-engines",
                "n_psms": df[engine_cols].all(axis=1).sum(),
            },
            ignore_index=True,
        )
    if "majority" not in df_num_psms["method"].values:
        n_majority_engines = np.ceil(0.5 * len(engine_cols))
        n_psms = (df[engine_cols].sum(axis=1) >= n_majority_engines).sum()
        df_num_psms = df_num_psms.append(
            {"method": "majority-engines", "n_psms": n_psms}, ignore_index=True
        )
    df_num_psms = df_num_psms.sort_values("n_psms", ascending=False)
    return df_num_psms


def get_num_psms_against_q_cut(
    df, methods, q_val_cut, initial_engine, all_engines_version
):
    """
    Returns a dataframe containing number of top target PSMs against q-value used as cut-off to identify top targets.
    Args:
        df (pd.DataFrame): dataframe containing results from search engines and ML training
        methods (List): list of methods to use. If None, use all methods
        q_val_cut (float): list of q-values to use, default is None (use values between 1e-4 and 1e-1)
        initial_engine (str): name of initial engine
        all_engines_version (List): List containing truncated engine names


    Returns:
        df (pd.DataFrame): dataframe containing number of PSMs at each q-value cut-off for each method
    """
    df = calc_all_final_q_vals(
        df, initial_engine=initial_engine, frac_tp=0.9, top_psm_only=True
    )

    # Get q-value list
    if q_val_cut is None:
        q_val_cut = sorted(
            [
                float(f"{i}e-{j}")
                for i in np.arange(1, 10)
                for j in np.arange(4, 1, -1)
            ]
        ) + [1e-1]

    # Get a list of methods
    # q-values from percolator are ignored, as these are not comparible to our results
    if methods is None:
        methods = [c for c in df.columns if "top_target" in c]
        methods = [c for c in methods if "ursgal" not in c and "from" not in c]
    # Initiate the dataframe for the results
    df_num_psms_q = pd.DataFrame(
        index=[str(q) for q in q_val_cut],
        columns=methods + ["top_target_any-engine", "top_target_all-engines"],
    )
    engine_cols = [f"top_target_{m}" for m in all_engines_version]
    for cut in q_val_cut:
        # Get the top-targets for this q_value_cut-off
        df = mark_top_targets(df, q_cut=cut)
        # Calculate the number of q-values for this cut-off
        df_num_psms = df[[c for c in df.columns if "top_target_" in c]].sum()
        df_num_psms_q.loc[str(cut), :] = df_num_psms[df_num_psms_q.columns]

        # Calculate results for any engine identifies a PSM as a top-target
        tmp = df[engine_cols].any(axis=1)
        df_num_psms_q.loc[str(cut), "top_target_any-engine"] = sum(tmp)
        # Calculate results for all-engines in agreement
        tmp = df[engine_cols].all(axis=1)
        df_num_psms_q.loc[str(cut), "top_target_all-engines"] = sum(tmp)
        # Calculate results for majorty-engines in agreement
        n_majority_engines = np.ceil(0.5 * len(engine_cols))
        tmp = df[engine_cols].sum(axis=1) >= n_majority_engines
        df_num_psms_q.loc[str(cut), "top_target_majority-engines"] = sum(tmp)
    return df_num_psms_q


def analyse(
    df_training,
    initial_engine,
    q_cut,
    frac_tp,
    top_psm_only,
    all_engines_version,
    plot_prefix,
    plot_dir,
    classifier,
):
    """
    Main function to analyse results.
    Args:
        df_training (pd.DataFrame): input dataframe
        initial_engine (str): name of initial engine
        q_cut (float): q-value to use as cut off
        frac_tp (float): estimate of fraction of true positives in target dataset
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        all_engines_version (List): List containing engine names and their respective version
        plot_prefix (str): output file prefix
        plot_dir (str): directory to save plots to
        classifier (str): name of classifier

    Returns:
        df_training (pd.DataFrame): dataframe with columns added for q-values, ranks and top target

    """
    # Get q-values
    df_training = calc_all_final_q_vals(
        df_training,
        initial_engine=initial_engine,
        frac_tp=frac_tp,
        top_psm_only=top_psm_only,
    )

    # Flag top targets
    df_training = mark_top_targets(df_training, q_cut)

    # Get ranks
    df_training = get_ranks(df_training)

    # for e1 in all_engines_version:
    #     df_new_top_targets, df_old_top_targets = get_shifted_psms(
    #         df_training, e1, classifier, n_return=None
    #     )
    #     df_new_top_targets.to_csv(
    #         plot_dir + f"{plot_prefix}_{classifier}_{e1}_new_top_targets.csv"
    #     )
    #     df_old_top_targets.to_csv(
    #         plot_dir + f"{plot_prefix}_{classifier}_{e1}_old_top_targets.csv"
    #     )

    return df_training
