import pytest
import re

from iterative_training_helpers import is_matching_filename, get_split_options


@pytest.mark.parametrize(
    "filename, pattern, value_map, expected",
    [
        (
            # all wildcards
            "something_a_b_c_1.raw",
            r".*_(?P<val1>[^_]+)_(?P<val2>[^_]+)_(?P<val3>[^_]+)_(?P<val4>[^_]+)\.raw$",
            {"val1": "*", "val2": "*", "val3": "*", "val4": "*"},
            True,
        ),
        (
            # different length
            "something_a_b.raw",
            r".*_(?P<val1>[^_]+)_(?P<val2>[^_]+)\.raw$",
            {"val1": "*", "val2": "*"},
            True,
        ),
        (
            # wrong character
            "something_a_b.raw",
            r".*_(?P<val1>[^_]+)_(?P<val2>[^_]+)\.raw$",
            {"val1": "*", "val2": "a"},
            False,
        ),
        (
            # lists
            "something_a_b.raw",
            r".*_(?P<val1>[^_]+)_(?P<val2>[^_]+)\.raw$",
            {"val1": ["a", "c"], "val2": ["b", "c"]},
            True,
        ),
        (
            # individual characters
            "something_1_c.raw",
            r".*_(?P<val1>[^_]+)_(?P<val2>[^_]+)\.raw$",
            {"val1": "1", "val2": "c"},
            True,
        ),
    ],
)
def test_is_matching_filename(filename, value_map, pattern, expected):
    re_pattern = re.compile(pattern)
    assert is_matching_filename(filename, pattern=re_pattern, **value_map) == expected

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
