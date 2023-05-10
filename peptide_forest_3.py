import peptide_forest
import multiprocessing as mp
import argparse

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="config path json")
    parser.add_argument("-o", help="output file")
    parser.add_argument("-mp_limit", help="max number of processes to use")
    args = parser.parse_args()

    pf = peptide_forest.PeptideForest(
        config_path=args.c,
        output=args.o,
        max_mp_count=args.mp_limit,
    )
    pf.prep_ursgal_csvs()
    pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()
