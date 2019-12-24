# left off here


def fit(
        df_training,
        classifier,
        n_train,
        n_eval,
        train_top_data,
        use_cross_validation,
        feature_cols,
        hyper_parameters,
        q_cut,
        q_cut_train,
):
    """
    Fit a given model to the data. Training data consists of targets with q-values < 1%
    and 50% of the decoys. Test data is all targets and the other 50% of the decoys.
    Validation data is targets/decoys from a different experiment (if provided).
    Args:
        df_training (pd.DataFrame): dataframe containing top target and top decoy for each spectrum
        classifier (str): which model to fit
        n_train (int): number of training iterations
        n_eval (int): number of of iterations to average score over at the end of the run
        train_top_data (bool): if True only train on top target/decoy per spectrum, if False train on all data
        use_cross_validation (bool): if True use cross validation
        feature_cols (list):
        hyper_parameters (Dict): dictionary containing hyper parameters for the model
        q_cut (float): cut off for q-values below which a target PSM is counted as top-target
        q_cut_train (float):    cut off for q-values below which a target PSM is counted as top-target,
                                when setting top-targets to train on
    Returns:
        (Tuple): consisting of:
            clfs (Dict):    dictionary containing a list of classifiers for each initial_score_col, that are fitted
                            to data at each iteration, with scalers for normalizing data
            psms (Dict):    dictionary of dictionaries containing number of PSMs identified for train, test and
                            validation data for each initial_score_col
            psms_avg (Dict): like psms but calculated on average score
            psms_engine (Dict): dictionary of dictionaries containing number of PSMs identified by search engine
                                for train, test and validation data, for each initial_score_col
            df (pd.DataFrame): input dataframe with score column added
            feature_importances (Dict): dictionary of dataframe containing feature importance for each initial_score_col
    """
    # fit_model_at or fit_model_cv