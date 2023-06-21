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
        config_path="./docker_test_data/config_local_full.json",  # args.c
        output=output,  # args.o,
        memory_limit=None,  # args.m,
        in_memory=True,
    )
    pf.boost(
        write_results=True,
        dump_train_test_data=False,
        eval_test_set=True,
        drop_used_spectra=False,
    )

    from peptide_forest.visualization import plot_q_value_curve
    from pathlib import Path

    filepath = list(pf.spectrum_index.keys())[0].split("/")[-1].split(".")[0]
    p = Path(output).parent / filepath / Path(output).name
    files = {"File": p}
    plot_q_value_curve(files)
