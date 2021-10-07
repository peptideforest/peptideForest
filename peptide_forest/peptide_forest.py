from pathlib import Path
import json
from peptide_forest.tools import Timer
import peptide_forest.knowledge_base
import peptide_forest.prep
import pandas as pd


class PeptideForest:
    def __init__(self, initial_engine, ursgal_path_dict, output):
        # Attributes
        self.ini_eng = initial_engine
        self.output = output
        with open(ursgal_path_dict) as json_file:
            self.ursgal_dict = json.load(json_file)

        self.input_df = None

    def _safe_write(self, path):
        config_path = Path(path)
        config_path.mkdir(parents=True, exist_ok=True)

    def prep_ursgal_csvs(self):
        engine_lvl_dfs = []

        # Retrieve list of columns shared across all files
        all_cols = []
        for file in self.ursgal_dict.keys():
            with open(file) as f:
                all_cols.append(set(f.readline().replace("\n", "").split(",")))
        shared_cols = list(
            set.intersection(*all_cols)
            - set(peptide_forest.knowledge_base.parameters["remove_cols"])
        )

        # Read in engines one by one
        for file, info in self.ursgal_dict.items():
            with Timer(description=f"Slurping in df for {info['engine']}"):
                df = pd.read_csv(file, usecols=shared_cols + [info["score_col"]])

                # Add information
                df["Engine"] = info["engine"]
                df["Score"] = df[info["score_col"]]

                # Drop irrelevant columns
                df.drop(columns=info["score_col"], inplace=True)

                # Check for duplicated rows
                init_len = len(df)
                df.drop_duplicates(inplace=True)
                rows_dropped = init_len - len(df)
                if rows_dropped != 0:
                    raise Warning(
                        f"{rows_dropped} duplicated rows were dropped in {file}."
                    )

                engine_lvl_dfs.append(df)

        combined_df = pd.concat(engine_lvl_dfs, sort=True).reset_index(drop=True)
        combined_df.fillna(
            {"Sequence Post AA": "-", "Sequence Pre AA": "-", "Modifications": "None"},
            inplace=True,
        )
        combined_df = combined_df.convert_dtypes()

        # Assert there are no overlaps between sequences in target and decoys
        if any(
            combined_df.groupby("Sequence").agg({"Is decoy": "nunique"})["Is decoy"]
            != 1
        ):
            raise ValueError("Target and decoy sequences overlap.")

        combined_df.drop(
            labels=combined_df[combined_df["Sequence"].str.contains("X") == True].index,
            axis=0,
            inplace=True,
        )

        if "mScore" in combined_df.columns:
            min_mscore = combined_df[combined_df["mScore"] != 0]["mScore"].min()
            combined_df.loc[combined_df["mScore"] == 0, "mScore"] = min_mscore

        self.input_df = combined_df

    def calc_features(self):
        peptide_forest.prep.calc_features(self.input_df)
