import numpy as np

class RollingWindow():

    def __init__(self, window_length):
        self._window_length = window_length
        self._values = []

    def add(self, value):
        self._values.append(value)
        if len(self._values) > self._window_length:
            self._values.pop(0)

        return self

    def size(self):
        return len(self._values)

    def mean(self):
        return np.nanmean(self._values)

    def max(self):
        return np.nanmax(self._values)

    def min(self):
        return np.nanmin(self._values)

    def std(self):
        return np.nanstd(self._values)