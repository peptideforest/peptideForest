"""Basic function to record execution time."""
import datetime
from collections import defaultdict
from typing import Dict, List

from loguru import logger


def convert_to_bytes(s):
    if s is None:
        return None

    units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    # Convert to lowercase
    s = s.lower()
    # Get the number part
    number = float(s[:-1])
    # Get the unit part
    unit = s[-1]
    # Check if the unit is valid
    if unit not in units:
        raise ValueError(f"Invalid unit {unit}")
    return int(number * units[unit])


def defaultdict_to_dict(d):
    """Recursively convert a defaultdict to a dict."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def count_elements_in_nested_dict(nested_dict: Dict[List]) -> int:
    """Count the total number of list elements in a dictionary with lists as values."""
    element_count = 0
    for _, list_of_elements in nested_dict.items():
        element_count += len(list_of_elements)
    return element_count


class Timer:
    """Basic class to use for timing/benchmarking steps of the code."""

    def __init__(self, description):
        """Initialize new timer object.

        Args:
            description (str): name of the timer that is to be printed once completed.
        """
        self.description = description
        self.start = datetime.datetime.now()
        self.context = False

    def _stop(self):
        """Calculate and log execution time."""
        dt = str(
            datetime.timedelta(
                seconds=(datetime.datetime.now() - self.start).total_seconds()
            )
        )
        logger.info(f"{self.description} in {dt}")

    def __del__(self):
        """Record deletion event."""
        if self.context is False:
            self._stop()

    def __enter__(self):
        """Record start event when entering context."""
        self.context = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record stop event upon leaving context."""
        self._stop()
