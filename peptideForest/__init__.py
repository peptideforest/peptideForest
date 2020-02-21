import logging

import peptideForest.runtime
import peptideForest.setup_dataset
import peptideForest.models
import peptideForest.results
import peptideForest.plot
import peptideForest.knowledge_base

logging_level_to_constants = {
    None: logging.DEBUG,
}

logging.basicConfig(
    format="peptide forest [%(asctime)s] - %(message)s", level=logging.DEBUG
)
