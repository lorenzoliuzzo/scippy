from .parametric_surface import ParametricSurface
from .interval import Interval 
import numpy as np


class Torus(ParametricSurface):
    # A torus in R^3.

    def __init__(self, R, r, centre):
        """ 
        Parameters
        ----------
        centre  The centre of the torus.
        R       The major radius of the torus.
        r       The minor radius of the torus.
        """
        
        self.centre = centre
        self.R = R
        self.r = r
        super().__init__(
            lambda u, v: self.centre[0] + (self.R + self.r * np.cos(v)) * np.cos(u),
            lambda u, v: self.centre[1] + (self.R + self.r * np.cos(v)) * np.sin(u),
            lambda u, v: self.centre[2] + self.r * np.sin(v),
            Interval(0, 2 * np.pi),
            Interval(0, 2 * np.pi)
        )

