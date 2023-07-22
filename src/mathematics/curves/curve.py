class Curve:
    # A curve is a function from R to R^n.


    def __init__(self, f, I):
        """
        Parameters
        ----------
        f : function
            A function from R to R^n.
        I : Interval
            The domain of the curve.
        """
        self.f = f
        self.domain = I

    def __call__(self, t):
        # Evaluate the curve at t.
        if self.domain.contains(t): # Check if t is in the domain of the curve.
            return self.f(t)
        raise ValueError(f"t = {t} is not in the domain of the curve.")
    
    

# class Helix(Curve):
#     """
#     A helix in R^3.
#     """

#     def __init__(self, radius, pitch):
#         self.radius = radius
#         self.pitch = pitch
#         super().__init__(lambda t: self.radius * [np.cos(t), np.sin(t), self.pitch * t])


# class Line(Curve):
#     """
#     A line in R^2.
#     """

#     def __init__(self, slope, intercept):
#         self.slope = slope
#         self.intercept = intercept
#         super().__init__(lambda t: self.slope * t + self.intercept)








# class ParametricCylinder(ParametricSurface):


#     def __init__(self, r, u, v):
#         super().__init__(
#             lambda u, v: r * np.cos(u),
#             lambda u, v: r * np.sin(u),
#             lambda u, v: v,
#             u, v
#         )


# class ParametricEllipsoid(ParametricSurface):


#     def __init__(self, a, b, c, u, v):
#         super().__init__(
#             lambda u, v: a * np.cos(u) * np.sin(v),
#             lambda u, v: b * np.sin(u) * np.sin(v),
#             lambda u, v: c * np.cos(v),
#             u, v
#         )


# class ParametricHyperboloid(ParametricSurface):

#     def __init__(self, a, b, c, u, v):
#         super().__init__(
#             lambda u, v: a * np.cosh(u) * np.cos(v),
#             lambda u, v: b * sinh(u) * np.cos(v),
#             lambda u, v: c * np.sin(v),
#             u, v
#         )


# class ParametricHyperbolicParaboloid(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * u,
#             lambda u, v: b * v,
#             lambda u, v: u ** 2 - v ** 2,
#             u, v
#         )


# class ParametricEllipticParaboloid(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * u,
#             lambda u, v: b * v,
#             lambda u, v: u ** 2 + v ** 2,
#             u, v
#         )


# class ParametricHyperbolicCylinder(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * np.cosh(u),
#             lambda u, v: b * sinh(u),
#             lambda u, v: v,
#             u, v
#         )


# class ParametricEllipticCylinder(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * np.cos(u),
#             lambda u, v: b * np.sin(u),
#             lambda u, v: v,
#             u, v
#         )


# class ParametricHyperbolicHelicoid(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * np.cosh(u) * np.cos(v),
#             lambda u, v: b * sinh(u) * np.cos(v),
#             lambda u, v: u,
#             u, v
#         )


# class ParametricEllipticHelicoid(ParametricSurface):

#     def __init__(self, a, b, u, v):
#         super().__init__(
#             lambda u, v: a * np.cos(u) * np.cos(v),
#             lambda u, v: b * np.sin(u) * np.cos(v),
#             lambda u, v: u,
#             u, v
#         )
