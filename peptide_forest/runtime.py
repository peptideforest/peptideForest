import time


class PFTimer(object):
    def __init__(self):
        self.times = {}
        self.was_stopped = set()

    def keys(self):
        return self.times.keys()

    def __getitem__(self, key):
        unit = (" sec", " min")
        if key not in self.was_stopped:
            if key in self.times.keys():
                elapsed = round((time.time() - self.times[key]), 2)
                if elapsed > 60:
                    elapsed = round(elapsed / 60, 2)
                    selected_unit = unit[1]
                else:
                    selected_unit = unit[0]
                self.times[key] = str(elapsed) + selected_unit
                self.was_stopped.add(key)
            else:
                self.times[key] = time.time()
        return self.times[key]
