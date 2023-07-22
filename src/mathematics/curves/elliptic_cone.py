from .parametric_surface import ParametricSurface


class EllipticCone(ParametricSurface):
    # An elliptic cone in R^3.

    def __init__(self, center, a, b, height):
        """
        Parameters
        ----------
        a       Scaling parameter in the x-direction.
        b       Scaling parameter in the y-direction.
        height  Height of the cone.
        center  The center point of the cone.
        """
        
        self.a = a
        self.b = b
        self.height = height
        self.center = center
        super().__init__(
            lambda u, v: self.center[0] + self.a * u * np.cos(v),
            lambda u, v: self.center[1] + self.b * u * np.sin(v),
            lambda u, v: self.center[2] + self.height * u
        )