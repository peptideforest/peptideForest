import datetime


class Timer:
    def __init__(self, description):
        self.description = description
        self.start = None

    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = str(
            datetime.timedelta(
                seconds=(datetime.datetime.now() - self.start).total_seconds()
            )
        )
        print(f"{self.description} in {dt}")
