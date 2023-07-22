import numpy as np
from mathematics import Node, Variable

class Interval:
    # An interval class to represent a subset of R.

    def __init__(self, start, end):
        """
        Parameters
        ----------
        start   The start of the interval.
        end     The end of the interval.
        """
        self.start = start
        self.end = end

    def contains(self, t):
        # Check if t is in the interval.
        if isinstance(t, np.ndarray):
            return np.logical_and(self.start <= t, t <= self.end)
        elif isinstance(t, Node):
            return np.logical_and(self.start <= t.value, t.value <= self.end)                
        else:
            return np.all(self.start <= t) and np.all(t <= self.end)