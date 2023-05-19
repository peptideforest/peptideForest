import argparse
import multiprocessing as mp

import peptide_forest
import peptide_forest.visualization

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
        config_path="./docker_test_data/config_local_full.json",  # args.c
        output="./docker_test_data/test_output.csv",  # args.o,
        memory_limit=None,  # args.m,
        max_mp_count=None,  # args.mp_limit,
    )
    pf.fit()
    peptide_forest.visualization.plot_model_performance(
        pf.training_performance,
        "Model Performance (random forest) with no training after 10 epochs",
    )
    pf.get_results()
    # pf.write_output()
