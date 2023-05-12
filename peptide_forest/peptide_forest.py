"""Main Peptide Forest class."""
import json
import random
import psutil

import pandas as pd
from loguru import logger

import peptide_forest.knowledge_base
import peptide_forest.prep
import peptide_forest.results
import peptide_forest.training
from peptide_forest.tools import Timer, convert_to_bytes


class PeptideForest:
    """Main class to handle peptide forest functionalities."""

    def __init__(self, config_path, output, memory_limit=None):
        """Initialize new peptide forest class object.
        
        Args:
            config_path (str): a path to a json file with configuration parameters
            output (str): output file path 
            initial_engine (str, None): sets initial scoring engine if engine name is given. defaults to None where the
                                        engine with most PSMs at q-cut is chosen.
        """ ""
        # Attributes
        self.output_path = output
        with open(config_path, "r") as json_file:
            self.params = json.load(json_file)
        self.init_eng = self.params.get("initial_engine", None)

        self.input_df = None
        self.timer = Timer(description="\nPeptide forest completed in")
        self.memory_limit = convert_to_bytes(memory_limit)
        self.max_chunk_size = None

    @staticmethod
    def _get_sample_lines(file, n_lines, sampled_lines=None):
        if n_lines is None:
            return None
        total_lines = sum(1 for l in open(file))
        skip_idx = random.sample(range(1, total_lines), total_lines - n_lines)
        return skip_idx

    def set_chunk_size(self, safety_margin=0.8):
        """Set max number of lines to be read per file."""
        self.prep_ursgal_csvs(n_lines=10)
        self.calc_features()
        n_files = len(self.params["input_files"])
        df_mem = self.input_df.memory_usage(deep=True).sum() / len(self.input_df)
        mem = psutil.virtual_memory()
        if mem.used > self.memory_limit:
            logger.warning(
                f"Memory limit of {self.memory_limit / 1024**2} Mb exceeded. "
                f"Used memory: {mem.used / 1024**2} Mb"
            )
            # todo: change to raise error ???
            if mem.available > self.memory_limit:
                free_mem = self.memory_limit
            # free_mem = mem.available
        else:
            free_mem = self.memory_limit - mem.used
        self.max_chunk_size = int(
            (free_mem) * safety_margin / df_mem / n_files
        )

    def get_data_chunk(self):
        """Get generator that yields data chunks for training."""
        self.prep_ursgal_csvs(n_lines=self.max_chunk_size)
        self.calc_features()
        yield self.input_df

    def prep_ursgal_csvs(self, n_lines: int = None):
        """Combine engine files named in ursgal dict and preprocesses dataframe for training."""
        engine_lvl_dfs = []

        # Retrieve list of columns shared across all files
        all_cols = []
        for file in self.params["input_files"].keys():
            with open(file, encoding="utf-8-sig") as f:
                all_cols.append(set(f.readline().replace("\n", "").split(",")))
        shared_cols = list(
            set.intersection(*all_cols)
            - set(peptide_forest.knowledge_base.parameters["remove_cols"])
        )

        # Read in engines one by one
        for file, info in self.params["input_files"].items():
            with Timer(description=f"Slurped in unified csv for {info['engine']}"):
                file_size = sum(1 for l in open(file))
                if file_size < n_lines:
                    logger.warning(
                        f"File {file} is too small to sample {n_lines} lines. Sampling {file_size} lines instead."
                    )
                    skip_idx = None
                else:
                    skip_idx = self._get_sample_lines(file, n_lines)
                df = pd.read_csv(
                    file, usecols=shared_cols + [info["score_col"]], skiprows=skip_idx
                )

                # Add information
                df["score"] = df[info["score_col"]]

                # Drop irrelevant columns
                df.drop(columns=info["score_col"], inplace=True)

                # Check for duplicated rows
                init_len = len(df)
                df.drop_duplicates(inplace=True)
                rows_dropped = init_len - len(df)
                if rows_dropped != 0:
                    logger.warning(
                        f"{rows_dropped} duplicated rows were dropped in {file}."
                    )

                engine_lvl_dfs.append(df)

        combined_df = pd.concat(engine_lvl_dfs, sort=True).reset_index(drop=True)
        combined_df.fillna(
            {"sequence_post_aa": "-", "sequence_pre_aa": "-", "modifications": "None"},
            inplace=True,
        )
        combined_df = combined_df.convert_dtypes()

        # Assert there are no overlaps between sequences in target and decoys
        shared_seq_target_decoy = (
            combined_df.groupby("sequence").agg({"is_decoy": "nunique"})["is_decoy"]
            != 1
        )
        if any(shared_seq_target_decoy):
            logger.warning("Target and decoy sequences overlap.")
            combined_df = combined_df.loc[
                ~combined_df["sequence"].isin(
                    shared_seq_target_decoy[shared_seq_target_decoy].index
                )
            ]

        combined_df.drop(
            labels=combined_df[combined_df["sequence"].str.contains("X") == True].index,
            axis=0,
            inplace=True,
        )

        if "m_score" in combined_df.columns:
            min_mscore = combined_df[combined_df["m_score"] != 0]["m_score"].min()
            combined_df.loc[combined_df["m_score"] == 0, "m_score"] = min_mscore

        self.input_df = combined_df

    def calc_features(self):
        """Calculate and adds features to dataframe."""
        logger.info("Calculating features...")
        with Timer("Computed features"):
            self.input_df = peptide_forest.prep.calc_row_features(self.input_df)
            self.input_df = peptide_forest.prep.calc_col_features(self.input_df)

    def fit(self):
        """Perform cross-validated training and evaluation."""

        self.input_df = self.get_data_chunk()

        with Timer(description="Trained model in"):
            (
                self.trained_df,
                self.feature_importances,
                self.n_psms,
            ) = peptide_forest.training.train(
                df=self.input_df,
                sensitivity=self.params.get("sensitivity", 0.9),
                q_cut=self.params.get("q_cut", 0.01),
                q_cut_train=self.params.get("q_cut_train", 0.10),
                n_train=self.params.get("n_train", 10),
                n_eval=self.params.get("n_eval", 10),
            )

    def get_results(self):
        """Interpret classifier output and appends final data to dataframe."""
        with Timer(description="Processed results in"):
            self.output_df = peptide_forest.results.process_final(
                df=self.trained_df,
                init_eng=self.init_eng,
                sensitivity=self.params.get("sensitivity", 0.9),
                q_cut=self.params.get("q_cut", 0.01),
            )
            self.output_df["modifications"].replace(
                {"None": None}, inplace=True, regex=False
            )

    def write_output(self):
        """Write final csv to file."""
        self.output_df.to_csv(self.output_path)
