#!/usr/bin/env python
"""Estimate q-values using machine learning methods
"""
import peptide_forest.setup_dataset as setup_dataset
import peptide_forest.analyse_results as analyse_results
import peptide_forest.fit_models as fit_models
import peptide_forest.plot_results as plot_results
from peptide_forest import prep

import itertools
import json
import multiprocessing
import pprint
import os
import time


import click
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)


def plot_annotated_spectra():
    """Summary.
    """
    pass

@click.command()
@click.option('--mindsai_csv' , '-m', help='e.g. original minds ai')
@click.option('--path_dict' , '-p', required=True, help='ursgal path dict')
@click.option('--output_file', '-o', required=False, help='tmp_peptideForest.csv')
# @click.option('--classifier', '-c', required=False, help='default RF')
# @click.option('--kernel', '-k', required=False, help='default None')
# @click.option('--n_train', '-nt', required=False, help=' default 5')
# @click.option('--n_eval', '-ne', required=False, help=' default 5')
# @click.option('--hyper_param_dict', '-h', required=False, help=' default None')
# @click.option('--train_top_data', '-t', required=False, help=' default True')
# @click.option('--use_crossvalidation', '-x', required=False, help=' default True')
# @click.option('--plot_dir', '-pd', required=False, help='./plots/')
# @click.option('--plot_prefix', '-pf', required=False, help='')
# @click.option('--plot_annotated_spectra', '-pa', required=False, help='default False')
# @click.option('--plot_rank_shift', '-pr', required=False, help='default True')
# @click.option('--initial_engine', '-i', required=False, help='msgfplus')
def main(
    path_dict=None,
    output_file='tmp_peptideForest.csv',
    mindsai_csv=None,
    classifier='RF-reg',
    kernel=None,
    n_train=5,
    n_eval=5,
    hyper_param_dict=None,
    train_top_data=True,
    use_crossvalidation=True,
    used_engines=None,
    plot_dir='./plots/',
    plot_prefix='',
    plot_annotated_spectra=False,
    plot_rank_shift=True,
    initial_engine='msgfplus'
):
    """
    Extract features from training set, impute missing values, fit model and
        predict.

    Args:
        path_dict (str): Description
        output_file (str): path to outputfile
        classifier (str, optional): name of the classifier
        kernel (str, optional): Description
        n_train (int, optional): Number of training iterations
        n_eval (int, optional): Number of evaluation iterations
        initial_score_col (str, optional): Name of the Score column to estimate top targets
        hyper_param_dict (dict, optional): hyper parameter for random forest
        train_top_data (bool, optional): Only use top data (0.1% q-value) for training
        use_crossvalidation (bool, optional): Whether or not to use crossvalidation
    """

    #
    # To be discussed ...
    #   - engines are trunctated somewhere on the way .. why ?
    #   - engines that have not assigned a spectrum to a given sequence get score == 0
    #       for that spectrum and it is counted as new top score ...
    #  - keep_ursgal is screwing classifier?
    #  - calc_final_q calues sets q-value to 1 if it has not been detected wth the search engine - not 0 anymore
    #       see analysis results 129
    #  - each fit_model returns the df_training, does that interfer with the results ?

    timer = PFTimer()
    timer['complete_run_time']

    default_hyper_parameters = {
        'RF' : {"random_state": 0, 'n_estimators': 100, 'max_depth': 22, 'max_features': 7,
                'n_jobs': multiprocessing.cpu_count() - 1},
        'SVM': {"random_state": 0, "max_iter": 50000},
        'RF-reg': {"random_state": 0, 'n_estimators': 100, 'max_depth': 22, 'max_features': 7,
                   'n_jobs': multiprocessing.cpu_count() - 1},
        'RF-reg.alt': {'n_estimators': 100, 'max_depth': 22, 'random_state': 0,
                       'class_weights': None, 'min_samples_split': 2, 'min_samples_leaf': 1,
                       'max_features': 7}
    }
    if hyper_param_dict is None:
        hyper_param_dict = default_hyper_parameters[classifier]
    print(f'Peptide Forest initialised with classifier: {classifier}')
    print('Using hyper parameters:')
    pprint.pprint(hyper_param_dict)

    dfs = []
    timer['reading_csv']
    forced_dtypes = {
        'Complies search criteria' : str,
        'Conflicting uparam' : str,
        'Substitutions': str
    }

    if mindsai_csv is not None:
        input_df = pd.read_csv(mindsai_csv, dtype=forced_dtypes)
        df_training, old_data, feature_cols = setup_dataset.make_dataset(
            input_df=input_df,
            combine_engines=True,
            keep_ursgal=False
        )
    else:
        path_dict = dict(json.loads(path_dict))
        input_df = setup_dataset.combine_ursgal_csv_files(path_dict)
        
        # setup_dataset.make_dataset
        df_training, old_data, feature_cols = setup_dataset.make_dataset(
            input_df=input_df,
            combine_engines=True,
            keep_ursgal=False,
        )

    if input_df.shape[0] < 100:
        raise Exception(
            '''Too few idents to run machine learning '''
            '''DataFrame has only {0} rows'''.format(input_df.shape[0])
        )


    all_engines = list(input_df['engine'].unique())
    all_engines_truncated = []
    for e in all_engines:
        x = e.split('_')
        all_engines_truncated.append(x[0])

    print('Working on results from engines {0} and {1}'.format(
        ', '.join(all_engines[:-1]),
        all_engines[-1]
        )
    )

    # leave in while testing, output dataframes
    df_training.to_csv(output_file.split('.csv')[0] + '-features.csv')
    input_df.to_csv(output_file)
    # TODO readd old cols
    
    print('Read csvs and preprocessed df in {reading_csv} min.'.format(**timer))
    #
    # Training on the Data
    #
    timer['fit_model']

    # set up dictionaries to store results and classifiers

    clss = {}  # store the trained classifiers
    PSMs = {}  # store the number of PSMs idendtified at each iteration
    PSMs_avg = {}  # store the number of PSMs idendtified at each iteration
    PSMs_engine_cv = {}  # store engine results: the engine used as the initial ordering of PSMs
    feature_importance = {}  # store dataframes containing the importance of the features used

    if f'Score_processed_{classifier}' in df_training.columns:
        df_training = df_training.drop(columns=[f'Score_processed_{classifier}'])

    model_type = f'{classifier} - all data - cv method - top only - from {initial_engine}'
    # ^--- needs to be updated as cv and top only are kwargs to this function.

    df_training['Is decoy'] = df_training['Is decoy'].astype(bool)
    clss = {}
    PSMs = {}
    PSMs_avg = {}

    clss[model_type], PSMs[model_type], \
        PSMs_avg[model_type], PSMs_engine_cv[initial_engine], \
        df_training, feature_importance[model_type] = fit_models.fit_model(
            classifier,
            feature_cols,
            df_training,
            n_train=n_train,
            n_eval=n_eval,
            initial_score_col='Score_processed_{0}'.format(initial_engine),
            hp_dict_in=hyper_param_dict,
            train_top_data=train_top_data,
            use_cv=use_crossvalidation,
            q_cut=0.01,
            q_cut_train=0.1
        )

    print('Fitted model in {fit_model} min'.format(**timer))
    print()
    print('Feature importance:')
    for score_col, df_feature_importance in feature_importance[model_type].items():
        print(score_col)
        print(df_feature_importance.head())

    #
    # Analysing results
    #
    timer['analysing_results']
    if os.path.exists(plot_dir) is False:
        os.mkdir(plot_dir)

    # get q-values
    df_training = analyse_results.calc_all_final_q_vals(df_training, from_method=initial_engine)
    # flag which are top targets
    df_training = analyse_results.get_top_targets(df_training, q_val_cut=0.01)
    # get the ranks
    df_training = analyse_results.get_ranks(df_training, from_scores=True)

    plot_results.plot_num_psms_by_method(
        df_training,
        output_file=os.path.join(
            plot_dir,
            f'{plot_prefix}_PSMs_by_method.pdf'
        ),
        print_values=True,
        show_plot=False
    )

    plot_results.plot_num_psms_against_q(
        df_training,
        output_file=os.path.join(
            plot_dir,
            f'{plot_prefix}_num_psms_vs_q.pdf'
        ),
        from_method='msgfplus'
    )

    # Using only top ranked PSM per spectrum

    cols = [c for c in df_training.columns if 'top_target' in c]
    df_training = df_training.drop(cols, axis=1)
    df_training = analyse_results.get_top_targets(df_training, q_val_cut=0.01)
    # print(df_training.columns)

    # plot the rank-rank plots
    for e1, e2 in itertools.permutations(all_engines_truncated + [classifier], 2):
        plot_results.plot_ranks(
            df_training,
            e1,
            e2,
            use_top_psm=True,
            n_psms=3,
            output_file=os.path.join(
                plot_dir,
                f'{plot_prefix}_{e1}_vs_{e2}.pdf'
            ),
            show_plot=False
        )
    for e1 in all_engines_truncated:
        df_new_top_targets, df_old_top_targets = analyse_results.get_shifted_psms(
            df_training,
            e1,
            classifier,
            n_return=None
        )
        df_new_top_targets.to_csv(plot_dir + f'{plot_prefix}_{classifier}_{e1}_new_top_targets.csv')
        df_old_top_targets.to_csv(plot_dir + f'{plot_prefix}_{classifier}_{e1}_old_top_target.csv')

    print('Analysed results in {analysing_results} min'.format(**timer))
    timer['writing_csv']
    df_training.to_csv(output_file, index=False)
    print('Wrote output csv {writing_csv} min'.format(**timer))
    print('Complete run time: {complete_run_time} min'.format(**timer))


class PFTimer(object):
    def __init__(self):
        self.times = {}
        self.was_stopped = set()

    def keys(self):
        return self.times.keys()

    def __getitem__(self, key):
        if key not in self.was_stopped:
            if key in self.times.keys():
                self.times[key] = round((time.time() - self.times[key]) / 60, 3)
                self.was_stopped.add(key)
            else:
                self.times[key] = time.time()
        return self.times[key]


if __name__ == '__main__':
    main()
    """
    p peptide_forest_1_0_0.py -p '{
            "/Users/matthew/mindsai/CellZome/data/new_experiment_data/Calwo_5ppm_04854_F1_R8_P0109699E13_TMT10_mascot_dat2csv_1_0_0_pmap_unified.csv": {
                "engine": "mascot",
                "score_col": "Mascot:Score"
            },
            "/Users/matthew/mindsai/CellZome/data/new_experiment_data/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_omssa_2_1_9_pmap_unified.csv": {
                "engine": "omssa",
                "score_col": "OMSSA:pvalue"
            },
            "/Users/matthew/mindsai/CellZome/data/new_experiment_data/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_msgfplus_v2018_06_28_pmap_unified.csv": {

                "engine": "msgfplus",
                "score_col": "MS-GF:SpecEValue"
            },
            "/Users/matthew/mindsai/CellZome/data/new_experiment_data/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_xtandem_vengeance_pmap_unified.csv": {
                "engine": "xtandem",
                "score_col": "X\\!Tandem:hyperscore"
            },
            "/Users/matthew/mindsai/CellZome/data/new_experiment_data/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_msfragger_20190222_pmap_unified.csv": {
                "engine": "msfragger",
                "score_col": "MSFragger:Hyperscore"
            }
        }'

    p peptide_forest_1_0_0.py -p '{
            "/Users/cf322940/data/peptideForest/E13/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_mascot_dat2csv_1_0_0_accepted_pmap_unified.csv": {
                "engine": "mascot",
                "score_col": "Mascot:Score"
            },
            "/Users/cf322940/data/peptideForest/E13/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_omssa_2_1_9_pmap_unified.csv": {
                "engine": "omssa",
                "score_col": "OMSSA:pvalue"
            },
            "/Users/cf322940/data/peptideForest/E13/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_msgfplus_v2018_06_28_pmap_unified.csv": {

                "engine": "msgfplus",
                "score_col": "MS-GF:SpecEValue"
            },
            "/Users/cf322940/data/peptideForest/E13/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_xtandem_vengeance_pmap_unified.csv": {
                "engine": "xtandem",
                "score_col": "X\\!Tandem:hyperscore"
            },
            "/Users/cf322940/data/peptideForest/E13/Offset_6_7ppm__04854_F1_R8_P0109699E13_TMT10_msfragger_20190222_pmap_unified.csv": {
                "engine": "msfragger",
                "score_col": "MSFragger:Hyperscore"
            }
        }'
    """
