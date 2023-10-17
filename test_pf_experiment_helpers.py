import pandas as pd
import pytest
from unittest.mock import patch, mock_open
import glob

from iterative_training_helpers import is_matching_filename
from pf4_paper_experiments import extract_variables, get_split_options


@pytest.mark.parametrize(
    "filename, strain, fraction, enzyme, rep, expected",
    [
        ("something_a_b_c_1", "*", "*", "*", "*", True),
        ("something_a_b_c_1", "*", "*", "c", "*", True),
        ("something_a_b_x_1", "*", "*", "c", "*", False),
        ("something_a_b_c_1", "*", ["a", "b"], ["c", "d"], "*", True),
        ("something_z_b_c_1", ["a", "b"], "*", ["c", "d"], "*", False),
        ("something_a_b_c_2", ["a", "x"], "*", "c", ["1", "2", "3"], True),
        ("something_a_b_c_4", ["a", "x"], "*", "c", ["1", "2", "3"], False),
    ],
)
def test_is_matching_filename(filename, strain, fraction, enzyme, rep, expected):
    assert is_matching_filename(filename, strain, fraction, enzyme, rep) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "prefix_something_a_b_c_1.raw",
            {"strain": "a", "fraction": "b", "enzyme": "c", "rep": "1"},
        ),
        (
            "anotherThing_x_y_z_2.raw",
            {"strain": "x", "fraction": "y", "enzyme": "z", "rep": "2"},
        ),
    ],
)
def test_extract_variables(filename, expected):
    assert extract_variables(filename) == expected


@pytest.fixture
def mock_csv_files(tmpdir):
    sample_data = [
        (
            {"raw_data_location": ["something_a_b_c_1.raw"], "other_column": [123]},
            "test1.csv",
        ),
        (
            {"raw_data_location": ["something_x_y_z_2.raw"], "other_column": [456]},
            "test2.csv",
        ),
        (
            {"raw_data_location": ["somethingelse_x_y_z_1.raw"], "other_column": [789]},
            "test3.csv",
        ),
    ]

    for data, name in sample_data:
        df = pd.DataFrame(data)
        csv_file = tmpdir.join(name)
        df.to_csv(csv_file, index=False)

    return tmpdir


def test_get_split_options(mock_csv_files):
    with patch(
        "glob.glob",
        return_value=[
            str(mock_csv_files.join("test1.csv")),
            str(mock_csv_files.join("test2.csv")),
        ],
    ):
        options = get_split_options(mock_csv_files)

    assert options["strain"] == {"a", "x"}
    assert options["fraction"] == {"b", "y"}
    assert options["enzyme"] == {"c", "z"}
    assert options["rep"] == {"1", "2"}
