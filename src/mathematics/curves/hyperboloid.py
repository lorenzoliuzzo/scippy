from .parametric_surface import ParametricSurface
from .interval import Interval

import numpy as np


class Hyperboloid(ParametricSurface):
    # A hyperboloid in R^3.

    def __init__(self, a, b, centre, interval_u):
        """
        Parameters
        ----------
        a       The major radius of the hyperboloid.
        b       The minor radius of the hyperboloid.
        centre  The centre of the hyperboloid.
        """
        
        self.centre = centre
        self.a = a
        self.b = b
        super().__init__(
            lambda u, v: self.centre[0] + self.a * np.cosh(u) * np.cos(v),
            lambda u, v: self.centre[1] + self.a * np.cosh(u) * np.sin(v),
            lambda u, v: self.centre[2] + self.b * np.sinh(u),
            interval_u,
            Interval(0, 2 * np.pi)
        )