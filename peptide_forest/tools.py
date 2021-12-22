"""Basic function to record execution time."""
import datetime

from loguru import logger


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
