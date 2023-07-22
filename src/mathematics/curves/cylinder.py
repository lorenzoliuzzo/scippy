from .parametric_surface import ParametricSurface


class Cylinder(ParametricSurface):
    # A cylinder in R^3.

    def __init__(self, r, h):
        """
        Parameters
        ----------
        r   The radius of the cylinder.
        h   The height of the cylinder.
        """

        super().__init__(
            lambda u, v: r * np.cos(u),
            lambda u, v: r * np.sin(u),
            lambda u, v: v
        )