from .parametric_surface import ParametricSurface


class EllipticParaboloid(ParametricSurface):
    # An elliptic paraboloid in R^3.

    def __init__(self, a, b, center):
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
            lambda u, v: self.center[2] + u ** 2
        )