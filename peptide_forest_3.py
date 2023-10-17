import peptide_forest
import multiprocessing as mp
import argparse

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="config path json")
    parser.add_argument("-o", help="output file")
    args = parser.parse_args()

    pf = peptide_forest.PeptideForest(
        config_path="pxd010824/config_new_decoys.json",  # args.c,
        output="test_output_universal_features_xgb.csv",  # args.o,
    )
    pf.prep_ursgal_csvs()
    pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()
