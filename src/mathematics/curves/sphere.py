from .parametric_surface import ParametricSurface
from .interval import Interval

import numpy as np


class Sphere(ParametricSurface):
    # A sphere in R^3.

    def __init__(self, radius, centre):
        """
        Parameters
        ----------
        radius  The radius of the sphere.
        centre  The centre of the sphere.
        """

        self.centre = centre
        self.radius = radius
        super().__init__(
            lambda u, v: self.centre[0] + self.radius * np.cos(u) * np.sin(v),
            lambda u, v: self.centre[1] + self.radius * np.sin(u) * np.sin(v),
            lambda u, v: self.centre[2] + self.radius * np.cos(v),
            Interval(0, 2 * np.pi),
            Interval(0, np.pi)
        )
