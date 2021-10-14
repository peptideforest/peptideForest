import json
import warnings
from loguru import logger

import pandas as pd

import peptide_forest.knowledge_base
import peptide_forest.prep
import peptide_forest.results
import peptide_forest.training
from peptide_forest.tools import Timer


class PeptideForest:
    def __init__(self, initial_engine, ursgal_path_dict, output):
        # Attributes
        self.init_eng = initial_engine
        self.output_path = output
        with open(ursgal_path_dict, "r") as json_file:
            self.ursgal_dict = json.load(json_file)

        self.input_df = None
        self.timer = Timer(description="\nPeptide forest completed in")

    def prep_ursgal_csvs(self):
        """
        Combines engine files named in ursgal dict and preprocesses dataframe for training.
        """
        engine_lvl_dfs = []

        # Retrieve list of columns shared across all files
        all_cols = []
        for file in self.ursgal_dict.keys():
            with open(file, encoding="utf-8-sig") as f:
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
        shared_seq_target_decoy = (
            combined_df.groupby("Sequence").agg({"Is decoy": "nunique"})["Is decoy"]
            != 1
        )
        if any(shared_seq_target_decoy):
            warnings.warn("Target and decoy sequences overlap.")
            combined_df = combined_df.loc[
                ~combined_df["Sequence"].isin(
                    shared_seq_target_decoy[shared_seq_target_decoy].index
                )
            ]

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
        """
        Calculates and adds features to dataframe
        """
        logger.info("Calculating features...")
        with Timer("Computed features"):
            self.input_df = peptide_forest.prep.calc_row_features(self.input_df)
            self.input_df = peptide_forest.prep.calc_col_features(self.input_df)

    def fit(self):
        """
        Performs cross-validated training and evaluation.
        """
        logger.info(f"Training from initial engine: {self.init_eng}")
        with Timer(description="Trained model in"):
            (
                self.trained_df,
                self.feature_importances,
                self.n_psms,
            ) = peptide_forest.training.train(
                df=self.input_df,
                init_eng=self.init_eng,
                sensitivity=0.9,
                q_cut=0.01,
                q_cut_train=0.10,
                n_train=10,
                n_eval=10,
            )

    def get_results(self):
        """
        Interprets classifier output and appends final data to dataframe.
        """
        with Timer(description="Processed results in"):
            self.output_df = peptide_forest.results.process_final(
                df=self.trained_df, init_eng=self.init_eng, sensitivity=0.9, q_cut=0.01
            )

    def write_output(self):
        """
        Writes final csv to file.
        """
        self.output_df.to_csv(self.output_path)
