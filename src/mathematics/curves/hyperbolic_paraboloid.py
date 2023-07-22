from .parametric_surface import ParametricSurface
from .interval import Interval

import numpy as np


class HyperbolicParaboloid(ParametricSurface):
    # A hyperbolic paraboloid in R^3.

    def __init__(self, a, b, center, interval_u, interval_v):
        """
        Parameters
        ----------
        a       Scaling parameter in the x-direction.
        b       Scaling parameter in the y-direction.
        center  The center point of the paraboloid.
        """
        self.a = a
        self.b = b
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * u * np.cos(v),
            lambda u, v: self.center[1] + self.b * u * np.sin(v),
            lambda u, v: self.center[2] + self.a * u ** 2 - self.b * v ** 2,
            interval_u,
            interval_v
        )
