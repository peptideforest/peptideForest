import peptide_forest
import multiprocessing as mp
import argparse

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="config path json")
    parser.add_argument("-o", help="output file")
    parser.add_argument("-m", help="memory limit")
    args = parser.parse_args()

    pf = peptide_forest.PeptideForest(
        # config_path=args.c,
        config_path="./docker_test_data/config_local.json",
        # output=args.o,
        output="./docker_test_data/output.csv",
        # memory_limit=args.m,
        memory_limit="100m",
    )
    pf.set_chunk_size()
    # pf.prep_ursgal_csvs()
    # pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()