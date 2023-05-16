import argparse
import multiprocessing as mp

import peptide_forest

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="config path json")
    parser.add_argument("-o", help="output file")
    parser.add_argument("-m", help="memory limit")
    parser.add_argument("-mp_limit", help="max number of processes to use")
    args = parser.parse_args()

    pf = peptide_forest.PeptideForest(
        config_path="./docker_test_data/config_local_full.json",#args.c
        output="./docker_test_data/test_output.csv",#args.o,
        memory_limit="250m",#args.m,
        max_mp_count=4,#args.mp_limit,
    )
    pf.set_chunk_size()
    # pf.prep_ursgal_csvs()
    # pf.calc_features()
    pf.fit()
    pf.plot_model_performance("Model Performance (random forest) with no training after 10 epochs")
    pf.get_results()
    # pf.write_output()
