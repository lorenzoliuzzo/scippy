from .parametric_curve import ParametricCurve
from .interval import Interval

import numpy as np


class Circle(ParametricCurve):
    # A circle in R^2.

    def __init__(self, radius, centre):
        """
        Parameters
        ----------
        radius  The radius of the circle.
        centre  The centre of the circle.
        """

        self.centre = centre
        self.radius = radius
        super().__init__(
            lambda t: self.centre[0] + self.radius * np.cos(t),
            lambda t: self.centre[1] + self.radius * np.sin(t),
            Interval(0, 2 * np.pi)
        )