import peptide_forest

pf = peptide_forest.PeptideForest(initial_engine="msgfplus_v2019_04_18", ursgal_path_dict="config/ursgal_path_dict.json", output="output/")
pf.prep_ursgal_csvs()
pf.calc_features()
