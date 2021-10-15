import peptide_forest
import multiprocessing as mp
import argparse

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='initial engine (optional)')
    parser.add_argument('-u', help='ursgal path dict json')
    parser.add_argument('-o', help='output file')
    args = parser.parse_args()

    pf = peptide_forest.PeptideForest(
        ursgal_path_dict=args.u,
        output=args.o,
        initial_engine=args.i,
    )
    pf.prep_ursgal_csvs()
    pf.calc_features()
    pf.fit()
    pf.get_results()
    pf.write_output()
