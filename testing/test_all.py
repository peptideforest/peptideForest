import json
import pandas as pd
import collections
import itertools
from peptideForest import setup_dataset, results


def test_combine_ursgal_csv_files():
    with open("data/config/ursgal_path_dict.json") as j:
        path_dict = json.load(j)
    j.close()

    test_combine_ursgal = setup_dataset.combine_ursgal_csv_files(path_dict=path_dict)
    reference_combine_ursgal = pd.read_pickle("ref_data/post_combine_ursgal.pkl")
    assert test_combine_ursgal.equals(reference_combine_ursgal)


def test_extract_features():

    input_df = pd.read_pickle("ref_data/post_combine_ursgal.pkl")

    test_df_training, test_old_cols, test_feature_cols = setup_dataset.extract_features(
        input_df
    )

    ref_old_cols = [
        "Accuracy (ppm)",
        "Charge",
        "Complies search criteria",
        "Conflicting uparam",
        "Exp m/z",
        "Is decoy",
        "MS-GF:DeNovoScore",
        "MS-GF:EValue",
        "MS-GF:RawScore",
        "MS-GF:SpecEValue",
        "MSFragger:Hyperscore",
        "MSFragger:Intercept of expectation model (expectation in log space)",
        "MSFragger:Matched fragment ions",
        "MSFragger:Neutral mass of peptide",
        "MSFragger:Next score",
        "MSFragger:Number of missed cleavages",
        "MSFragger:Number of tryptic termini",
        "MSFragger:Precursor neutral mass (Da)",
        "MSFragger:Slope of expectation model (expectation in log space)",
        "MSFragger:Total possible number of matched theoretical fragment ions",
        "Mascot:Score",
        "Modifications",
        "OMSSA:evalue",
        "OMSSA:pvalue",
        "Protein ID",
        "Retention Time (s)",
        "Score",
        "Sequence",
        "Sequence Post AA",
        "Sequence Pre AA",
        "Sequence Start",
        "Sequence Stop",
        "Spectrum ID",
        "Unnamed: 0",
        "X\!Tandem:expect",
        "X\!Tandem:hyperscore",
        "engine",
        "rank",
        "uCalc m/z",
    ]

    ref_feature_cols = [
        "PepLen",
        "lnNumPep",
        "Score_processed_xtandem",
        "CountProt",
        "Score_processed_msgfplus",
        "ln abs delta m/z",
        "Charge5",
        "delta_score_2_msgfplus",
        "delta_score_3_msgfplus",
        "Score_processed_mascot",
        "Mass",
        "Charge2",
        "Score_processed_omssa",
        "Charge3",
        "Charge4",
        "enzN",
        "enzInt",
        "delta m/z",
        "Score_processed_msfragger",
        "abs delta m/z",
    ]
    ref_df_training = pd.read_pickle("ref_data/post_extract_features.pkl")

    equal_df_training = test_df_training.equals(ref_df_training)
    equal_old_cols = set(test_old_cols) == set(ref_old_cols)
    equal_feature_cols = set(test_feature_cols) == set(ref_feature_cols)

    assert all([equal_df_training, equal_old_cols, equal_feature_cols])


def test_analysis():

    df_training = pd.read_pickle("ref_data/post_fit_model.pkl")

    df_training = results.basic(
        df_training,
        initial_engine="msgfplus",
        q_cut=0.01,
        from_scores=True,
        frac_tp=0.9,
        top_psm_only=True,
    )
    ref_df_training = pd.read_pickle("ref_data/post_analysis.pkl")

    # Since in find_psms_to_keep one decoy is dropped at random if both PSMs happen to be decoys
    # [TRISTAN] only appears to happen for mascot and RF-reg?
    q_val_cols = [col for col in df_training.columns if "q-value_" in col]
    for col in q_val_cols:
        if collections.Counter(df_training[col].to_list()) == collections.Counter(
            ref_df_training[col].to_list()
        ):
            df_training = df_training.drop(col, axis=1)
            ref_df_training = ref_df_training.drop(col, axis=1)

    assert df_training.equals(ref_df_training)


def test_get_num_psms_by_method():

    df = pd.read_pickle("ref_data/post_analysis.pkl")
    df = results.get_num_psms_by_method(df, methods=None)

    ref_df = pd.read_pickle("ref_data/ref_num_psms_by_method.pkl")

    assert ref_df.equals(df)


def test_get_num_psms_against_q():

    df = pd.read_pickle("ref_data/post_analysis.pkl")
    df = results.get_num_psms_against_q_cut(
        df, methods=None, q_val_cut=None, initial_engine="msgfplus"
    )

    ref_df = pd.read_pickle("ref_data/ref_num_psms_against_q.pkl")

    assert ref_df.equals(df)


def test_get_ranks():

    df = pd.read_pickle("ref_data/post_analysis.pkl")
    equality = []
    engines = ["mascot", "omssa", "msgfplus", "xtandem", "msfragger", "RF-reg"]

    for e1, e2 in itertools.permutations(engines, 2):
        df_plot = df.sort_values(
            f"Score_processed_{e2}", ascending=False
        ).drop_duplicates("Spectrum ID")
        df_plot = results.get_ranks(df_plot, from_scores=True, method="first")
        ref_df_plot = pd.read_pickle(f"ref_data/ref_get_ranks_{e1}_{e2}.pkl")
        equality.append(df_plot.equals(ref_df_plot))

    assert all(equality)


def test_get_shifted_psms():

    df_training = pd.read_pickle("ref_data/post_analysis.pkl")
    engines = ["mascot", "omssa", "msgfplus", "xtandem", "msfragger"]
    equality = []

    for e in engines:
        df_new_top_targets, df_old_top_targets = results.get_shifted_psms(
            df_training, x_name=e, y_name="RF-reg", n_return=None
        )
        ref_new_top_targets = pd.read_pickle(
            f"ref_data/ref_shifted_psms_RF-reg_{e}_new.pkl"
        )
        ref_old_top_targets = pd.read_pickle(
            f"ref_data/ref_shifted_psms_RF-reg_{e}_old.pkl"
        )
        equality.append(
            ref_new_top_targets.equals(df_new_top_targets)
            and ref_old_top_targets.equals(df_old_top_targets)
        )

    assert all(equality)


def test_all():
    test_combine_ursgal_csv_files()
    test_extract_features()
    test_analysis()
    test_get_num_psms_by_method()
    test_get_num_psms_against_q()
    test_get_ranks()
    test_get_shifted_psms()
