import peptide_forest
import multiprocessing as mp


def run_peptide_forest(config_path, output):
    mp.freeze_support()
    mp.set_start_method("fork", force=True)

    pf = peptide_forest.PeptideForest(
        config_path=config_path,
        output=output,
    )
    pf.prep_ursgal_csvs()
    pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()
