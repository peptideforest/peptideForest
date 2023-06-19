import argparse
import multiprocessing as mp
from uuid import uuid4

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
    output = f"./docker_test_data/{uuid4()}_output.csv"
    pf = peptide_forest.PeptideForest(
        config_path="./docker_test_data/config_local.json",  # args.c
        output=output,  # args.o,
        memory_limit=None,  # args.m,
    )
    pf.boost()
