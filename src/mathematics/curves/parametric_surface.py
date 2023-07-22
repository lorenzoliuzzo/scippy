import numpy as np
from .interval import Interval

class ParametricSurface:

    def __init__(self, x, y, z, domain_u, domain_v):
        """
        Parameters
        ----------
        x : function
            A function representing the x-coordinate of the surface.
        y : function
            A function representing the y-coordinate of the surface.
        z : function
            A function representing the z-coordinate of the surface.
        domain_u : Interval
            The closed interval for parameter u.
        domain_v : Interval
            The closed interval for parameter v.
        """
        self.x = x
        self.y = y
        self.z = z
        self.domain_u = domain_u
        self.domain_v = domain_v


    def __call__(self, u, v):
        """
        Evaluate the parametric surface at the given parameters u and v.

        Parameters
        ----------
        u : float
            The parameter value for u.
        v : float
            The parameter value for v.

        Returns
        -------
        numpy.array
            An array representing the (x, y, z) coordinates of the surface at (u, v).

        Raises
        ------
        ValueError
            If u or v is not within the specified domains.
        """
        # if np.isscalar(u):
        #     u = np.array([u])
        # if np.isscalar(v):
        #     v = np.array([v])

        if not self.domain_u.contains(u):
            raise ValueError(f"u value {u} is outside the domain [{self.domain_u.start}, {self.domain_u.end}]")

        if not self.domain_v.contains(v):
            raise ValueError(f"v value {v} is outside the domain [{self.domain_v.start}, {self.domain_v.end}]")
        
        return np.array([self.x(u, v), self.y(u, v), self.z(u, v)])
