from .autodiff import Variable
from .gradients import gradients
import numpy as np

class Curve:
    """
    A curve is a function from R to R^n.

    Parameters
    ----------
    t : Variable
        The independent variable of the curve.
    f : function
        The function that maps t to R^n.
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, t):
        return self.f(t)
    

class Circle(Curve):
    """
    A circle in R^2.
    """

    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius
        super().__init__(lambda t: self.centre + self.radius * [np.cos(t), np.sin(t)])


class Helix(Curve):
    """
    A helix in R^3.
    """

    def __init__(self, radius, pitch):
        self.radius = radius
        self.pitch = pitch
        super().__init__(lambda t: self.radius * [np.cos(t), np.sin(t), self.pitch * t])


class Line(Curve):
    """
    A line in R^2.
    """

    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
        super().__init__(lambda t: self.slope * t + self.intercept)


class ParametricCurve(Curve):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__(lambda t: [self.x(t), self.y(t)])


class ParametricSurface:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __call__(self, u, v):
        return self.x(u, v), self.y(u, v), self.z(u, v)

# class ParametricTorus(ParametricSurface):


#     def __init__(self, R, r, u, v):
#         super().__init__(
#             lambda u, v: (R + r * np.cos(v)) * np.cos(u),
#             lambda u, v: (R + r * np.cos(v)) * np.sin(u),
#             lambda u, v: r * np.sin(v),
#             u, v
#         )


class ParametricSphere(ParametricSurface):


    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius
        super().__init__(
            lambda u, v: self.centre.value[0] + self.radius * np.cos(u) * np.sin(v),
            lambda u, v: self.centre.value[1] + self.radius * np.sin(u) * np.sin(v),
            lambda u, v: self.centre.value[2] + self.radius * np.cos(v),
        )


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
