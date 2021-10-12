import datetime


class Timer:
    """
    Basic class to use for timing/benchmarking steps of the code.
    """

    def __init__(self, description):
        self.description = description
        self.start = datetime.datetime.now()
        self.context = False

    def _stop(self):
        dt = str(
            datetime.timedelta(
                seconds=(datetime.datetime.now() - self.start).total_seconds()
            )
        )
        print(f"{self.description} in {dt}")

    def __del__(self):
        if self.context is False:
            self._stop()

    def __enter__(self):
        self.context = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
