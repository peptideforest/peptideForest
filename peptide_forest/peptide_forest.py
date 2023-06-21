"""Main Peptide Forest class."""
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

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

    def __init__(
            self, config_path, output, memory_limit=None, max_mp_count=None,
            in_memory=False
    ):
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
        self.in_memory = in_memory

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

        if max_mp_count is not None:
            logger.warning("max_mp_count is deprecated. Use n_jobs in config instead.")

        self.timer = peptide_forest.tools.Timer(
            description="\nPeptide forest completed in"
        )
        self.config = PFConfig(self.params.get("config", {}))
        # todo: add default config to knowledge base
        # todo: remove max_mp_count and only use config. Redundancy is not necessary.
        if self.config.n_jobs.value is None:
            self.config.n_jobs.value = mp.cpu_count() - 1
        if self.config.n_jobs.value > mp.cpu_count():
            logger.warning(
                f"n_jobs ({self.config.n_jobs.value}) > cpu_count ({mp.cpu_count()}). "
                f"Setting n_jobs to {mp.cpu_count() - 1}."
            )
            self.config.n_jobs.value = mp.cpu_count() - 1
        if type(self.config.n_jobs.value) is not int:
            logger.warning(
                f"n_jobs ({self.config.n_jobs.value}) is not an integer. "
                f"Setting n_jobs to {mp.cpu_count() - 1}."
            )
            self.config.n_jobs.value = mp.cpu_count() - 1
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
            df = self.prep_ursgal_csvs(sample_dict=sample_dict)
            df = self.calc_features(df)
            n_files = len(self.params["input_files"])
            df_mem = df.memory_usage(deep=True).sum() / len(self.input_df)
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
            if not drop:
                ref = reference_spectra.copy()
            else:
                ref = reference_spectra
            (
                sample_dict,
                sampled_spectra,
            ) = peptide_forest.sample.generate_sample_dict(
                self.spectrum_index.copy(),
                file,
                reference_spectra_ids=ref,
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

            logger.info(f"Sampling {len(sampled_spectra)} spectra.")
            df = self.prep_ursgal_csvs(sample_dict=sample_dict)
            df = self.calc_features(df=df)
            yield df

    def get_data_cunk_from_df(
            self,
            df,
            reference_spectra=None,
            n_spectra=None,
            drop=True,
    ):
        random.seed(42)
        while True:
            if n_spectra is None:
                n_spectra = len(reference_spectra)
            if len(reference_spectra) == 0:
                logger.info("No more spectra to sample. Exiting.")
                break
            refs = random.sample(reference_spectra, n_spectra)
            if drop is True:
                reference_spectra = [r for r in reference_spectra if r not in refs]
            refs = [int(r) for r in refs]
            df = df[df["spectrum_id"].isin(refs)]
            logger.info(f"Sampling {len(refs)} spectra.")
            yield df

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

        return combined_df

    def calc_features(self, df):
        """Calculate and adds features to dataframe."""
        logger.info("Calculating features...")
        with peptide_forest.tools.Timer("Computed features"):
            df = peptide_forest.prep.calc_row_features(
                df, max_mp_count=self.config.n_jobs.value
            )
            df = peptide_forest.prep.calc_col_features(
                df, max_mp_count=self.config.n_jobs.value
            )
            return df

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

    def score_with_model(self, gen=None):
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

            yield df
        return None

    def get_results(self, gen=None, write_output=True):
        """Interpret classifier output and appends final data to dataframe."""
        with peptide_forest.tools.Timer(description="Processed results in"):
            scored_gen = self.score_with_model(gen=gen)
            temp_path = self.output_path.parent / "temp.csv"
            score_collection = []

            while True:
                try:
                    df = next(scored_gen)
                    df["modifications"].replace(
                        {"None": None}, inplace=True, regex=False
                    )
                except StopIteration:
                    break

                # check if output path exists
                # todo: note, this does not overwrite the output file if no new path is
                #   given, could be unexpected
                if os.path.exists(temp_path):
                    mode, header = "a", False
                else:
                    mode, header = "w", True
                if write_output:
                    df.to_csv(temp_path, mode=mode, header=header, index=False)
                    score_cols = [c for c in df.columns if "score_processed_" in c]
                    score_collection.append(
                        df[["spectrum_id", "sequence", "is_decoy", *score_cols]]
                    )
                    del df
                else:
                    score_collection.append(df)

            total_scores = pd.concat(score_collection).reset_index(drop=True)
            del score_collection

            # process results
            output_df = peptide_forest.results.process_final(
                df=total_scores,
                init_eng=self.init_eng,
                sensitivity=self.params.get("sensitivity", 0.9),
                q_cut=self.params.get("q_cut", 0.01),
            )
            del total_scores

            if write_output:
                with pd.read_csv(temp_path, chunksize=self.max_chunk_size) as reader:
                    for chunk in reader:
                        chunk = chunk.merge(
                            output_df.copy(),
                            how="left",
                            on=["spectrum_id", "sequence"],
                        )
                        # check if output path exists
                        # todo: note, this does not overwrite the output file if no new path is
                        #   given, could be unexpected
                        if os.path.exists(self.output_path):
                            mode, header = "a", False
                        else:
                            mode, header = "w", True
                        chunk.to_csv(
                            self.output_path, mode=mode, header=header, index=False
                        )
                os.remove(temp_path)
                return None
            else:
                return output_df

    def boost(
            self,
            write_results=True,
            dump_train_test_data=False,
            eval_test_set=True,
            retrain=False,
            drop_used_spectra=True,
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

            if self.in_memory is True:
                df = self.prep_ursgal_csvs()
                df = df[df["raw_data_location"] == file]
                df = self.calc_features(df)

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

                if self.in_memory is True:
                    self.input_df = self.get_data_cunk_from_df(
                        df.copy(), train_spectra, drop=drop_used_spectra
                    )
                else:
                    self.input_df = self.get_data_chunk(
                        file=self.file,
                        reference_spectra=train_spectra,
                        n_spectra=self.config.n_spectra.value,
                        drop=drop_used_spectra,
                    )

                if not retrain:
                    self.engine = None

                self.fit(fold=fold)

                if eval_test_set:
                    if self.in_memory is True:
                        eval_gen = self.get_data_cunk_from_df(df.copy(), test_spectra)
                    else:
                        eval_gen = self.get_data_chunk(
                            file=self.file, reference_spectra=test_spectra
                        )

                    _ = self.get_results(gen=eval_gen, write_output=write_results)

                logger.info(self.config)
                self.fold_configs[fold] = self.config
                fold += 1
