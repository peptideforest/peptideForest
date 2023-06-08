"""Main Peptide Forest class."""
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from uuid import uuid4

import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import peptide_forest.file_handling
import peptide_forest.knowledge_base
import peptide_forest.prep
import peptide_forest.results
import peptide_forest.sample
import peptide_forest.tools
import peptide_forest.training
from peptide_forest.pf_config import PFConfig


class PeptideForest:
    """Main class to handle peptide forest functionalities."""

    def __init__(self, config_path, output, memory_limit=None, max_mp_count=None):
        """Initialize new peptide forest class object.
        
        Args:
            config_path (str): a path to a json file with configuration parameters
            output (str): output file path 
            memory_limit (int, str): memory limit for peptide forest in format 1g, 1m ..
            max_mp_count (int): maximum number of processes to be used
        """ ""
        # Attributes
        with open(config_path, "r") as json_file:
            self.params = json.load(json_file)
        self.output_path = Path(output)
        self.memory_limit = peptide_forest.tools.convert_to_bytes(memory_limit)
        self.init_eng = self.params.get("initial_engine", None)

        self.input_df = None
        self.max_chunk_size = None
        self.engine = None
        self.scaler = None
        self.file = None
        self.unique_spectrum_ids = None
        self.spectrum_index = peptide_forest.sample.generate_spectrum_index(
            self.params["input_files"].keys()
        )
        self.buffered_scores = []

        engine_path = self.params.get("engine_path", None)
        if engine_path is not None:
            self.engine = peptide_forest.training.get_classifier()
            self.engine.load_model(engine_path)

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

        self.timer = peptide_forest.tools.Timer(
            description="\nPeptide forest completed in"
        )
        self.config = PFConfig(self.params.get("config", {}))
        # todo: add default config to knowledge base
        self.config.n_jobs.value = self.max_mp_count
        self.initial_config = self.config.copy()
        self.fold_configs = {}

    def set_chunk_size(self, safety_margin=0.8):
        """Set max number of lines to be read per file."""
        if self.memory_limit is None:
            logger.info("No memory limit set. Using default chunk size.")
            # todo: determine default / optimal chunk size if no max is given.
            self.max_chunk_size = 1e12
        else:
            sample_dict, sampled_spectra = peptide_forest.sample.generate_sample_dict(
                index_dict=self.spectrum_index,
                file=self.file,
                n_spectra=100,
            )
            self.prep_ursgal_csvs(sample_dict=sample_dict)
            self.calc_features()
            n_files = len(self.params["input_files"])
            df_mem = self.input_df.memory_usage(deep=True).sum() / len(self.input_df)
            self.max_chunk_size = int(
                self.memory_limit * safety_margin / df_mem / n_files
            )

    def get_data_chunk(
            self,
            file,
            reference_spectra=None,
            n_spectra=None,
            drop=True,
    ):
        """Get generator that yields data chunks for training."""
        # ensure reproducibility
        random.seed(42)
        self.spectrum_index = peptide_forest.sample.generate_spectrum_index(
            self.params["input_files"].keys()
        )

        while True:
            (
                sample_dict,
                sampled_spectra,
            ) = peptide_forest.sample.generate_sample_dict(
                self.spectrum_index.copy(),
                file,
                reference_spectra_ids=reference_spectra,
                n_spectra=n_spectra,
                max_chunk_size=self.max_chunk_size,
            )

            if drop is True:
                spectra = self.spectrum_index[file]
                spectra = {k: v for k, v in spectra.items() if k not in sampled_spectra}
                self.spectrum_index[file] = spectra
                if len(sampled_spectra) == 0:
                    logger.info("No more spectra to sample. Exiting.")
                    break

            logger.info(f"Sampling {n_spectra} spectra.")
            self.prep_ursgal_csvs(sample_dict=sample_dict)
            self.calc_features()
            yield self.input_df

    def prep_ursgal_csvs(self, sample_dict=None):
        """Combine engine files named in ursgal dict and preprocesses dataframe for
        training.
        """
        engine_lvl_dfs = []

        # Get shared columns
        shared_cols = peptide_forest.file_handling.shared_columns(
            self.params["input_files"].keys()
        )

        # Read in engines one by one
        for file, info in self.params["input_files"].items():
            df = peptide_forest.file_handling.load_csv_with_sampling_information(
                file,
                shared_cols + [info["score_col"]],
                sample_dict=sample_dict,
            )

            if df is None:
                continue

            df["score"] = df[info["score_col"]]
            df.drop(columns=info["score_col"], inplace=True)

            df = peptide_forest.file_handling.drop_duplicates_with_log(df, file)

            engine_lvl_dfs.append(df)

        combined_df = pd.concat(engine_lvl_dfs, sort=True).reset_index(drop=True)
        combined_df.fillna(
            {"sequence_post_aa": "-", "sequence_pre_aa": "-", "modifications": "None"},
            inplace=True,
        )
        combined_df = combined_df.convert_dtypes()

        peptide_forest.prep.check_target_decoy_sequence_overlap(combined_df)

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
        with peptide_forest.tools.Timer("Computed features"):
            self.input_df = peptide_forest.prep.calc_row_features(
                self.input_df, max_mp_count=self.max_mp_count
            )
            self.input_df = peptide_forest.prep.calc_col_features(
                self.input_df, max_mp_count=self.max_mp_count
            )

    def fit(self, fold=None):
        """Perform cross-validated training and evaluation."""
        if self.params.get("save_models", False):
            peptide_forest.file_handling.create_dir_if_not_exists(
                self.output_path.parent, "models"
            )

        with peptide_forest.tools.Timer(description="Trained model in"):
            (
                self.trained_df,
                self.feature_importances,
                self.n_psms,
                self.engine,
                self.scaler,
                self.config,
            ) = peptide_forest.training.train(
                gen=self.input_df,
                sensitivity=self.params.get("sensitivity", 0.9),
                config=self.config,
                model=self.engine,
                fold=fold,
                save_models=self.params.get("save_models", False),
                save_models_path=self.output_path.parent / "models",
            )

            # todo: dump prepared data for scoring results

    def score_with_model(self, gen=None, use_disk=False):
        if gen is None:
            gen = self.get_data_chunk(file=self.file)
        while True:
            # iterative loading of data
            try:
                df = next(gen)
            except StopIteration:
                break

            # predict scores
            feature_columns = peptide_forest.training.get_feature_columns(df)
            if self.scaler is None:
                self.scaler = StandardScaler().fit(df.loc[:, feature_columns])
            df.loc[:, feature_columns] = self.scaler.transform(
                df.loc[:, feature_columns]
            )

            df[f"score_processed_peptide_forest"] = self.engine.score_psms(
                df[feature_columns].astype(float)
            )

            if use_disk:
                p = Path(f"./temp/{uuid4()}.pkl")
                df.to_pickle(p)
                self.buffered_scores.append(p)
            else:
                # todo: assumes whole df can be returned
                return df
        return None

    def get_results(self, gen=None, use_disk=False, write_output=True):
        """Interpret classifier output and appends final data to dataframe."""
        with peptide_forest.tools.Timer(description="Processed results in"):
            df = self.score_with_model(gen=gen, use_disk=use_disk)
            if use_disk:
                # todo: only works in memory this way
                df = pd.concat([pd.read_pickle(p) for p in self.buffered_scores])
                self.buffered_scores = []

            # process results
            output_df = peptide_forest.results.process_final(
                df=df,
                init_eng=self.init_eng,
                sensitivity=self.params.get("sensitivity", 0.9),
                q_cut=self.params.get("q_cut", 0.01),
            )

            # check if output path exists
            # todo: note, this does not overwrite the output file if no new path is
            #   given, could be unexpected
            if os.path.exists(self.output_path):
                mode, header = "a", False
            else:
                mode, header = "w", True
            if write_output:
                output_df.to_csv(
                    self.output_path, mode=mode, header=header, index=False
                )
            return output_df

    def boost(
            self,
            write_results=True,
            dump_train_test_data=False,
            eval_test_set=True,
            retrain=False,
    ):
        """Perform cross-validated training and evaluation."""
        files = list(self.spectrum_index.keys())
        if len(files) > 1:
            logger.info("multiple files found in input, analyzing file by file.")

        for file in files:
            self.file = file
            logger.info(f"Analyzing file {file}...")
            logger.info("Determining max. chunk size...")
            self.set_chunk_size()

            # reset output path with filename as folder
            peptide_forest.file_handling.create_dir_if_not_exists(
                self.output_path.parent, Path(self.file).stem
            )
            self.output_path = (
                    self.output_path.parent / Path(
                self.file).stem / self.output_path.name
            )

            self.unique_spectrum_ids = list(
                dict.fromkeys(self.spectrum_index[file].keys())
            )

            if dump_train_test_data:
                peptide_forest.file_handling.create_dir_if_not_exists(
                    self.output_path.parent, "tt_data"
                )

            # create folds
            num_folds = self.config.n_folds.value
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            splits = cv.split(self.unique_spectrum_ids.copy())
            fold = 1
            for train_ids, test_ids in splits:
                unique_spectrum_ids = self.unique_spectrum_ids.copy()
                self.config = self.initial_config.copy()
                train_spectra = [unique_spectrum_ids[i] for i in train_ids]
                test_spectra = [unique_spectrum_ids[i] for i in test_ids]
                logger.info(
                    f"Starting Fold {fold} of {num_folds}: train on {len(train_spectra)} "
                    f"spectra, score {len(test_spectra)} spectra"
                )

                if dump_train_test_data:
                    with open(
                            self.output_path.parent / "tt_data" / f"data_f{fold}.json",
                            "w",
                    ) as f:
                        tt_data = {
                            "train_spectra": train_spectra,
                            "test_spectra": test_spectra,
                            "fold": fold,
                        }
                        json.dump(tt_data, f)

                self.input_df = self.get_data_chunk(
                    file=self.file,
                    reference_spectra=train_spectra,
                    n_spectra=self.config.n_spectra.value,
                )

                if not retrain:
                    self.engine = None

                self.fit(fold=fold)

                if eval_test_set:
                    eval_gen = self.get_data_chunk(
                        file=self.file, reference_spectra=test_spectra
                    )

                    if write_results:
                        write_output = True
                    else:
                        write_output = False

                    _ = self.get_results(
                        gen=eval_gen, use_disk=False, write_output=write_output
                    )

                logger.info(self.config)
                self.fold_configs[fold] = self.config
                fold += 1
