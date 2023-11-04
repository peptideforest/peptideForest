import pandas as pd

import peptide_forest
import multiprocessing as mp
from pathlib import Path
import os


def run_peptide_forest(config_path, output, buffer_calculated_df=False):
    mp.freeze_support()
    mp.set_start_method("fork", force=True)

    pf = peptide_forest.PeptideForest(
        config_path=config_path,
        output=output,
    )

    config_name = Path(config_path).name
    buffer_path = "./temp/" + config_name.replace("config_", "temp_df").replace(
        ".json", ".pkl"
    )

    if os.path.exists(buffer_path):
        pf.input_df = pd.read_pickle(buffer_path)
    else:
        pf.prep_ursgal_csvs()
        pf.calc_features()
        if buffer_calculated_df:
            pf.input_df.to_pickle(buffer_path)

    pf.fit()
    pf.get_results()
    pf.write_output()
