import numpy as np

class ParametricCurve:
    # A parametric curve in R^2.

    def __init__(self, x, y, I):
        """
        Parameters
        ----------
        x : function
            A function representing the x-coordinate of the curve.
        y : function
            A function representing the y-coordinate of the curve.
        domain : Interval
            The domain of the curve.
        """
        self.x = x
        self.y = y
        self.domain = I


    def __call__(self, t):
        """
        Evaluate the parametric curve at the given parameter t.

        Parameters
        ----------
        t : float
            The parameter value at which to evaluate the curve.

        Returns
        -------
        numpy.array
            An array representing the (x, y) coordinates of the curve at t.

        Raises
        ------
        ValueError
            If t is not within the specified domain.
        """
        if self.domain.contains(t):
            return np.array([self.x(t), self.y(t)])
        raise ValueError("t is not in the domain of the curve.")