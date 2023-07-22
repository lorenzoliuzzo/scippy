from .parametric_surface import ParametricSurface
from .interval import Interval

import numpy as np


class Cone(ParametricSurface):

    def __init__(self, apex, base_radius, height):
        """
        A cone in R^3.

        Parameters
        ----------
        apex : Quantity (array-like)
            The apex (tip) of the cone (x, y, z).
        base_radius : Quantity
            The radius of the base of the cone.
        height : Quantity
            The height of the cone.
        """
        self.apex = apex
        self.base_radius = base_radius
        self.height = height
        super().__init__(
            lambda u, v: self.apex[0] + u * self.base_radius * np.cos(v),
            lambda u, v: self.apex[1] + u * self.base_radius * np.sin(v),
            lambda u, v: self.apex[2] + u * self.height,
            Interval(0, 1),
            Interval(0, 2 * np.pi)
        )