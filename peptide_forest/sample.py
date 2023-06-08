"""Generating indices to sample from input data."""
import random
from collections import defaultdict

import peptide_forest.tools


def generate_spectrum_index(input_files):
    """Generate spectrum index for all input files.

    Format of the index is:
        {raw_data_location: {spectrum_id: {filename: [line_idx]}}}
    """
    spectrum_index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in input_files:
        with open(filename, "r", encoding="utf-8-sig") as file:
            header = next(file).strip().split(",")
            raw_data_location_idx = header.index("raw_data_location")
            spectrum_id_idx = header.index("spectrum_id")

            for i, line in enumerate(file):
                data = line.strip().split(",")
                raw_data_location = data[raw_data_location_idx]
                spectrum_id = data[spectrum_id_idx]

                spectrum_index[raw_data_location][spectrum_id][filename].append(i + 1)

    return peptide_forest.tools.defaultdict_to_dict(spectrum_index)


def generate_sample_dict(
        index_dict, reference_spectra_ids=None, n_spectra=None, max_chunk_size=None
):
    """Generate a sample dict to get all data lines for a given number of spectra.

    Args:
        index_dict (dict): dictionary indexing locations of spectrum ids in input files
        reference_spectra_ids (list, None): list of reference spectra ids, used for crossvalidation
        n_spectra (int, None): number of spectra to sample
        max_chunk_size (int, None): maximum number of lines to sample

    Returns:
        sample_dict (dict): dictionary of lines to keep per input file
        sampled_spectra (list): list of sampled spectrum ids
    """
    # todo: hack, fix
    first_file = list(index_dict.keys())[0]
    if reference_spectra_ids is not None:
        spectra_ids = reference_spectra_ids
    else:
        spectra_ids = list(index_dict[first_file].keys())
    # spectra_ids = [s for s in spectra_ids if s in reference_spectra_ids]
    if n_spectra is None:
        n_spectra = len(spectra_ids)
    sample_dict = defaultdict(list)
    sampled_lines = 0
    sampled_spectra = list()
    while sampled_lines <= max_chunk_size and len(sampled_spectra) < n_spectra:
        spectrum_id = random.choice(spectra_ids)
        sampled_spectra.append(spectrum_id)
        spectra_ids.remove(spectrum_id)
        spectrum_info = index_dict[first_file][spectrum_id]
        sampled_lines += peptide_forest.tools.count_elements_in_nested_dict(
            spectrum_info
        )
        for filename, line_idxs in spectrum_info.items():
            sample_dict[filename].extend(line_idxs)
    return dict(sample_dict), sampled_spectra
