import pytest
import re

from iterative_training_helpers import (
    is_matching_filename,
    get_split_options,
    generate_accepted_groups_dict,
)


@pytest.mark.parametrize(
    "filename, pattern, allowed_values, expected",
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
def test_is_matching_filename(filename, allowed_values, pattern, expected):
    re_pattern = re.compile(pattern)
    assert (
        is_matching_filename(
            filename, pattern=re_pattern, allowed_values=allowed_values
        )
        == expected
    )


def test_get_split_options():
    directory = "./test_files"

    pattern = re.compile(r"(?P<group1>[a-zA-Z0-9]+)_?(?P<group2>[a-zA-Z0-9]+)?")
    result = get_split_options(directory, pattern, col="colA")

    # Assuming you know the expected outcome based on your sample files
    expected = {"group1": {"value1", "value2"}, "group2": {"valueA", "valueB"}}

    assert result == expected


@pytest.mark.parametrize(
    "splitting_group, available_options, expected",
    [
        (
            "a",
            {"a": {1, 2, 3}, "b": {1, 2, 3}},
            [{"a": 1, "b": "*"}, {"a": 2, "b": "*"}, {"a": 3, "b": "*"}],
        )
    ],
)
def test_generate_accepted_groups_dict(splitting_group, available_options, expected):
    all_groups_dicts = [
        group_dict
        for group_dict in generate_accepted_groups_dict(
            splitting_group, available_options
        )
    ]
    assert all_groups_dicts == expected
