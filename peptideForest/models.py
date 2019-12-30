from peptideForest import prep
from peptideForest import classifier


def get_train_test_sets(
    df, use_cross_validation,
):
    """
    Split dataframe into multiple sets for training.
    Args:
        df (pd.DataFrame): experiment dataframe
        use_cross_validation (bool): if True use 3-fold cross-validation, if False train on all targets

    Returns:
        training_data (pd.DataFrame):   list containing data splits. If use_cross_validation is True contains has
                                        three cv splits. If use_cross_validation is False, contains train split
                                        (all targets, 50% decoys) and test split (all targets, other 50% decoys)
    """
    # get_q_vals


def get_q_vals(
    df, score_col, frac_tp, top_psm_only, get_fdr,
):
    """
    Calculate q-value for each PSM based on the score in the column score_col
    Args:
        df (pd.DataFrame): experiment dataframe
        score_col (str): name of column to rank PSMs by
        frac_tp (float): estimate of fraction of true positives in target dataset
        # [TRISTAN] ist fix true maybe remove?
        top_psm_only (bool): keep only highest scoring PSM for each spectrum
        # [TRISTAN] ist fix false maybe remove?
        get_fdr (bool): if True return the false detection rate as q-value, else return q-values

    Returns:
        df_scores (pd.DataFrame): dataframe containing q-values, score and whether PSM is target or decoy
    """


def find_psms_to_keep(df, score_col):
    """
    Flag PSMs to remove from q-value calculations, when the highest score for a given spectrum is shared by
    two or more PSMs.
    Args:
        df (pd.DataFrame): dataframe containing search engine scores for all PSMs
        score_col (str): score column to use when comparing PSM scores

    Returns:
        df (pd.DataFrame): updated dataframe with flag 'keep in': False means remove when calculating q-values

    """


def calc_num_psms(
    df, score_col, q_cut, frac_tp,
):
    """
    Calculate the number of PSMs with q-value < 1%, using the following criteria:
        - only the highest ranked PSMs for each spectra are considered
        - redundant (i.e. duplicate) peptide sequences are discarded
        - only target PSMs are counted
    Args:
        df (pd.DataFrame): dataframe containing scores for each PSM
        score_col (str): name of column to rank PSMs by
        q_cut (float): cut off for q-values below which a target PSM is counted as top-target
        frac_tp (float): estimate of fraction of true positives in target dataset

    Returns:
        n_psms (int): number of PSMs with q-value < 1% that match the above criteria

    """


def get_train_set(
    df, score_col, q_cut, frac_tp, train_top_data, sample_frac,
):
    """
    Return training dataset containing sample of decoys and all target PSMs with q-value less than 1%
    Args:
        df (pd.DataFrame): dataframe containing experiment data
        score_col (str): name of column containing score to rank PSMs by
        q_cut (float): cut off for q-values below which a target PSM is counted as a top-target
        frac_tp (float): estimate of fraction of true positives in target dataset
        train_top_data (bool): if False train on all the data, True only train on top target/decoy per spectrum
        # [TRISTAN] if to be included muss das noch nach oben vererbt werden
        sample_frac (float): ratio of decoy PSMs to target and decoy PSMs

    Returns:
        train (pd.DataFrame): dataframe containing selected target and decoy PSMs
    """
    # prep.get_top_target_decoy(df, score_col=score_col)


# This function was moved here from setup_dataset
def replace_missing_data_cv(
    train_j, train_k, test,
):
    """
    Replace missing feature values.
    Args:
        train_j (pd.DataFrames): dataframe containing feature values for training split
        train_k (pd.DataFrames): dataframe containing feature values for training split
        test (pd.DataFrames): dataframe containing feature values for test split

    Returns:
        train_j (pd.DataFrames): input dataframes with missing values filled in
        train_k (pd.DataFrames): input dataframes with missing values filled in
        test (pd.DataFrames): input dataframes with missing values filled in

    """


def get_training_model(
    training_type, hp_dict_in,
):
    """
    Select the learning model to use for training.
    Args:
        training_type (str): which type of model to use options include:
            - 'RF': 'Random Forest',
            - 'RF-reg': 'Random Forest - regression',
            - 'GBT': 'Gradient Boosted Trees',
            - 'SVM': 'Support Vector Machine',
            - 'GNB': 'Gaussian Naive Bayes classifier',
            - 'KNN': 'K Nearest Neighbors',
            - 'LR': 'Logistic Regression',
            - 'ADA': 'Ada Boosted Classifier with Decision Trees'
        hp_dict_in (Dict):  dictionary containing hyperparameters for the model, which are the same
                            as the scikit-learn parameters.

    Returns:
        clf (Any): selected classifier with hyperparameters and scoring_func function added
    """

    def score_psm(
        clf, data,
    ):
        """
        [TRISTAN] find out how this docstring is filled
        Args:
            clf:
            data:

        Returns:

        """
        # Lorem ipsum

    # clf.score_psm = types.MethodType(score_psm, clf)


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
    frac_tp,
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
        frac_tp (float): estimate of fraction of true positives in target dataset
        # [TRISTAN] frac_tp könnte hier (wenn gewollt) default value 0.9 bekommen genauso wie die beiden q_cuts
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
    # fit_model_at or fit_model_cv in now one combined function
