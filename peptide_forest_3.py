import peptide_forest
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)

    pf = peptide_forest.PeptideForest(
        initial_engine="msgfplus_v2018_06_28",
        ursgal_path_dict="config/ursgal_path_dict.json",
        output="peptide_forest_output.csv",
    )
    pf.prep_ursgal_csvs()
    pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()
