from .parametric_surface import ParametricSurface
from .interval import Interval

import numpy as np


class Helix(ParametricSurface):
    def __init__(self, radius, pitch, num_turns, center, axis=(0, 0, 1), initial_angle=0):
        """
        Parameters
        ----------
        radius : Quantity
            The radius of the helix.
        pitch : Quantity
            The distance between consecutive turns along the helix axis.
        num_turns : float
            The number of turns of the helix.
        center : Quantity (array-like)
            The center of the helix (x, y, z).
        axis : tuple of floats, optional
            The axis of the helix in 3D space. Default is (0, 0, 1), which corresponds to the z-axis.
        initial_angle : float, optional
            The initial angle of the helix in radians. Default is 0.
        """
        self.radius = radius
        self.pitch = pitch
        self.num_turns = num_turns
        self.center = center
        self.axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
        self.initial_angle = initial_angle

        super().__init__(
            lambda u, v: self.center[0] + self.radius * np.cos(self.initial_angle + u),
            lambda u, v: self.center[1] + self.radius * np.cos(self.initial_angle + u),
            lambda u, v: self.center[2] + self.pitch * v / (2 * np.pi),
            Interval(0, self.num_turns * 2 * np.pi),
            Interval(0, 2 * np.pi)
        )
