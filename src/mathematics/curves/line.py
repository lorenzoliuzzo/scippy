from .parametric_curve import ParametricCurve
from .interval import Interval

import numpy as np


class Line(ParametricCurve):
    # A line in R^n.

    def __init__(self, start, end):
        """
        Parameters
        ----------
        start   The start of the line.
        end     The end of the line.
        """

        self.start = start
        self.end = end
        super().__init__(
            lambda t: self.start + (self.end - self.start) * t,
            Interval(0, 1)
        )