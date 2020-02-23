import logging

import peptide_forest.runtime
import peptide_forest.setup_dataset
import peptide_forest.models
import peptide_forest.results
import peptide_forest.plot
import peptide_forest.knowledge_base

logging_level_to_constants = {
    None: logging.DEBUG,
}

logging.basicConfig(
    format="peptide forest [%(asctime)s] - %(message)s", level=logging.DEBUG
)
