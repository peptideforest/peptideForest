"""Process results for trained model."""
from peptide_forest.training import calc_q_vals


def process_final(
    df,
    init_eng,
    sensitivity,
    q_cut,
):
    """Add final outputs to dataframe the classifier was trained with.

    Args:
        df (pd.DataFrame): input data
        init_eng (str): initial engine to rank results by
        sensitivity (float): proportion of positive results to true positives in the data
        q_cut (float): q-value cutoff for PSM selection

    Returns:
        df (pd.DataFrame): final dataframe for output to csv
    """
    score_cols = [c for c in df.columns if "score_processed_" in c]
    for score_col in score_cols:
        # Get all final q-values
        df_scores = calc_q_vals(
            df=df,
            score_col=score_col,
            sensitivity=sensitivity,
            top_psm_only=True,
            get_fdr=True,
            init_score_col=init_eng,
        )
        q_col = f"q-value_{score_col.split('score_processed_')[-1]}"
        df[q_col] = 1.0
        df.loc[df_scores.index, q_col] = df_scores["q-value"]
        df[q_col].fillna(df_scores["q-value"].max(), inplace=True)

        # Mark top targets
        top_target_col = f"top_target_{score_col.split('score_processed_')[-1]}"
        df_top = df[(df[q_col] <= q_cut) & (~df["is_decoy"])]
        df_top = df_top.sort_values(q_col).drop_duplicates("spectrum_id")
        df[top_target_col] = False
        df.loc[df_top.index, top_target_col] = True

        # Rank targets
        rank_col = f"rank_{score_col.split('score_processed_')[-1]}"
        df[rank_col] = df[score_col].rank(ascending=False, method="first")

    return df
