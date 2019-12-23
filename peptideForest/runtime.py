import time


class PFTimer(object):
    def __init__(self):
        self.times = {}
        self.was_stopped = set()

    def keys(self):
        return self.times.keys()

    def __getitem__(self, key):
        if key not in self.was_stopped:
            if key in self.times.keys():
                self.times[key] = round((time.time() - self.times[key]) / 60, 3)
                self.was_stopped.add(key)
            else:
                self.times[key] = time.time()
        return self.times[key]