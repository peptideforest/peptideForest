"""Main Peptide Forest class."""
import json
import multiprocessing as mp
import random
from collections import defaultdict

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

import peptide_forest.knowledge_base
import peptide_forest.prep
import peptide_forest.results
import peptide_forest.training
from peptide_forest.tools import Timer, convert_to_bytes


class PeptideForest:
    """Main class to handle peptide forest functionalities."""

    def __init__(self, config_path, output, memory_limit=None, max_mp_count=None):
        """Initialize new peptide forest class object.
        
        Args:
            config_path (str): a path to a json file with configuration parameters
            output (str): output file path 
            memory_limit (int, str): memory limit for peptide forest in format 1g, 1m etc.
            max_mp_count (int): maximum number of processes to be used
        """ ""
        # Attributes
        self.output_path = output
        with open(config_path, "r") as json_file:
            self.params = json.load(json_file)
        self.init_eng = self.params.get("initial_engine", None)

        self.input_df = None
        self.timer = Timer(description="\nPeptide forest completed in")
        if max_mp_count is None:
            self.max_mp_count = mp.cpu_count() - 1
        else:
            try:
                self.max_mp_count = int(max_mp_count)
            except ValueError:
                logger.error(
                    "Invalid input for max_mp_count. Using available cores - 1."
                )
                self.max_mp_count = mp.cpu_count() - 1

        self.memory_limit = convert_to_bytes(memory_limit)
        self.max_chunk_size = None
        self.engine = None
        self.scaler = None
        self.training_performance = None
        self.spectrum_index = {}

    @staticmethod
    def _get_sample_lines(file, n_lines, sampled_lines=None):
        if n_lines is None:
            return None
        total_lines = sum(1 for l in open(file))
        skip_idx = random.sample(range(1, total_lines), total_lines - n_lines)
        return skip_idx

    def generate_spectrum_index(self):
        """Generate spectrum index for all input files.

        Format of the index is:
            {raw_data_location: {spectrum_id: {filename: [line_idx]}}}
        """
        self.spectrum_index = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for filename in self.params["input_files"]:
            with open(filename, "r", encoding="utf-8-sig") as file:
                header = next(file).strip().split(",")
                raw_data_location_idx = header.index("raw_data_location")
                spectrum_id_idx = header.index("spectrum_id")

                for i, line in enumerate(file):
                    data = line.strip().split(",")
                    raw_data_location = data[raw_data_location_idx]
                    spectrum_id = data[spectrum_id_idx]

                    self.spectrum_index[raw_data_location][spectrum_id][
                        filename
                    ].append(i)

    def set_chunk_size(self, safety_margin=0.8):
        """Set max number of lines to be read per file."""
        self.prep_ursgal_csvs(n_lines=10)
        self.calc_features()
        n_files = len(self.params["input_files"])
        df_mem = self.input_df.memory_usage(deep=True).sum() / len(self.input_df)
        self.max_chunk_size = int(self.memory_limit * safety_margin / df_mem / n_files)

    def get_data_chunk(self, mode="random", n_lines=None):
        """Get generator that yields data chunks for training."""
        if n_lines is None:
            n_lines = self.max_chunk_size

        if mode == "spectrum":
            # todo: implement
            pass
        elif mode == "drop":
            # todo: implement
            pass
        elif mode == "random":
            while True:
                self.prep_ursgal_csvs(n_lines=n_lines)
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
            self.input_df = peptide_forest.prep.calc_row_features(
                self.input_df, max_mp_count=self.max_mp_count
            )
            self.input_df = peptide_forest.prep.calc_col_features(
                self.input_df, max_mp_count=self.max_mp_count
            )

    def fit(self):
        """Perform cross-validated training and evaluation."""

        self.input_df = self.get_data_chunk()

        with Timer(description="Trained model in"):
            (
                self.trained_df,
                self.feature_importances,
                self.n_psms,
                self.engine,
                self.training_performance,
            ) = peptide_forest.training.train(
                gen=self.input_df,
                sensitivity=self.params.get("sensitivity", 0.9),
                q_cut=self.params.get("q_cut", 0.01),
                q_cut_train=self.params.get("q_cut_train", 0.10),
                n_train=self.params.get("n_train", 10),
                algorithm=self.params.get("algorithm", "random_forest_scikit"),
                max_mp_count=self.max_mp_count,
            )

    def get_results(self):
        """Interpret classifier output and appends final data to dataframe."""
        with Timer(description="Processed results in"):
            gen = self.get_data_chunk(mode="spectrum")
            iterations = 0
            while True:
                # iterative loading of data
                try:
                    df = next(gen)
                except StopIteration:
                    break

                # predict scores
                feature_columns = peptide_forest.training.get_feature_columns(df)
                # todo: store scaler after training and use it here
                if self.scaler is None:
                    self.scaler = StandardScaler().fit(df.loc[:, feature_columns])
                df.loc[:, feature_columns] = self.scaler.transform(
                    df.loc[:, feature_columns]
                )
                df[
                    f"score_processed_{self.params.get('algorithm', 'random_forest_scikit')}"
                ] = self.engine.predict(df[feature_columns])

                # process results
                output_df = peptide_forest.results.process_final(
                    df=df,
                    init_eng=self.init_eng,
                    sensitivity=self.params.get("sensitivity", 0.9),
                    q_cut=self.params.get("q_cut", 0.01),
                )

                # write results
                if iterations == 0:
                    output_df.to_csv(self.output_path, mode="w", header=True, index=False)
                else:
                    output_df.to_csv(self.output_path, mode="a", header=False, index=False)
                del output_df
                iterations += 1
