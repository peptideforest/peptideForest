import pytest
import re

from iterative_training_helpers import (
    is_matching_filename,
    get_split_options,
    generate_accepted_groups_dict,
    check_for_trained_models,
    get_model_name_str,
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


@pytest.mark.parametrize(
    "model_name, model_dir, config_dict, availability",
    [
        ("iamamodel", "./test_files/models", {"conf": {"model_type": "xgboost"}}, True),
        (
            "iamadifferentmodel",
            "./test_files/models",
            {"conf": {"model_type": "random_forest"}},
            True,
        ),
        ("iamamodel", "./test_files/models", {"not_conf": ""}, True),
        (
            "youwontfnidme",
            "./test_files/models",
            {"conf": {"model_type": "xgboost"}},
            False,
        ),
        (
            "iamadifferentmodel",
            "./test_files/models",
            None,
            True,
        ),
    ],
)
def test_check_for_trained_models(model_name, model_dir, config_dict, availability):
    assert check_for_trained_models(model_name, model_dir, config_dict) == availability


@pytest.mark.parametrize(
    "training_schedule, current_idx, expected_str",
    [
        (
            ["config_a.json", "config_b.json", "config_c.json", "config_d.json"],
            2,
            "model_a>b>c",
        ),
        (
            ["config_a.json", "config_b.json", "config_c.json", "config_d.json"],
            0,
            "model_a",
        ),
        (
            ["config_a.json", "config_b.json", "config_c.json", "config_d.json"],
            -1,
            None,
        ),
    ],
)
def test_get_model_name_str(training_schedule, current_idx, expected_str):
    assert get_model_name_str(training_schedule, current_idx) == expected_str
