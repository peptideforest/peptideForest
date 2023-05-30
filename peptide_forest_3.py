import argparse
import multiprocessing as mp

import peptide_forest
from peptide_forest.visualization import plot_model_performance, plot_q_value_curve, plot_psms_at_qval_threshold

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
        config_path="./docker_test_data/config_local.json",  # args.c
        output="./docker_test_data/test_output.csv",  # args.o,
        memory_limit=None,  # args.m,
        max_mp_count=None,  # args.mp_limit,
    )
    pf.boost()
    # pf.fit()
    # pf.get_results()
    files = {"Test": "./docker_test_data/test_output.csv"}
    title = "Test"

    plot_q_value_curve(files, title=title)
    plot_psms_at_qval_threshold(files, title=title)
    # pf.write_output()
