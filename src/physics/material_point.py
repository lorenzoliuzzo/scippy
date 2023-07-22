from physics import Quantity
from physics import units as U
from mathematics import Variable
# from mathematics.curves import ParametricSurface

import numpy as np


class MaterialPoint:
    """
    A material point in space.
    """

    def __init__(self, mass, 
                 position : Quantity = Quantity(np.array([0, 0, 0]), U.m),
                 velocity : Quantity = Quantity(np.array([0, 0, 0]), U.m / U.s)):
        """
        A material point in space.

        Parameters
        ----------
        mass : Quantity
            The mass of the material point.
        position : Quantity (array-like)
            The position of the material point (x, y, z).
        velocity : Quantity (array-like)
            The velocity of the material point (dx/dt, dy/dt, dz/dt).
        """
        self.mass = mass
        self.position = Variable('x', position)
        self.velocity = Variable(r'$\dot{x}$', velocity)

    def __str__(self):
        return f"MaterialPoint: mass: {self.mass}, position: {self.position}, velocity: {self.velocity}"


    def momentum(self):
        return self.mass * self.velocity
    
    def kinetic_energy(self):
        return self.mass * self.velocity**2 / 2
    

    # def constrain_to_surface(self, surface: ParametricSurface, tolerance: float = 1e-6, max_iterations: int = 100):
    #     """
    #     Constrain the material point to the given surface.

    #     Parameters
    #     ----------
    #     surface : ParametricSurface
    #         The surface to which the material point should be constrained.
    #     tolerance : float, optional
    #         A tolerance value for the constraint satisfaction. Default is 1e-6.
    #     max_iterations : int, optional
    #         Maximum number of iterations for the Newton-Raphson method. Default is 100.
    #     """
    #     for _ in range(max_iterations):
    #         # Evaluate the surface function and its derivatives at the current position of the material point.
    #         surface_position = surface(self.position.value[0], self.position.value[1])
    #         surface_normal = surface.normal(self.position.value[0], self.position.value[1])
            
    #         # Calculate the distance between the material point and the closest point on the surface.
    #         distance = np.linalg.norm(surface_position - self.position.value)

    #         # Check if the constraint is satisfied within the tolerance.
    #         if distance < tolerance:
    #             break

    #         # Update the position of the material point using the Newton-Raphson method.
    #         jacobian = surface.jacobian(self.position.value[0], self.position.value[1])
    #         delta_position = np.linalg.solve(jacobian.T @ jacobian, jacobian.T @ surface_normal)
    #         self.position.set_value(self.position.value - delta_position)


# class ConstrainedMaterialPoint(MaterialPoint):
#     def __init__(self, mass, position, velocity):
#         super().__init__(mass, position, velocity)
#         self.constraints = []

#     def add_constraint(self, constraint_curve):
#         """
#         Add a constraint represented by a ParametricCurve2D to the material point.

#         Parameters
#         ----------
#         constraint_curve : ParametricCurve2D
#             The constraint curve that restricts the movement of the material point.
#         """
#         if not isinstance(constraint_curve, ParametricCurve2D):
#             raise ValueError("Constraint must be represented by a ParametricCurve2D.")
#         self.constraints.append(constraint_curve)

#     def is_constrained(self, position):
#         """
#         Check if the position of the material point satisfies all constraints.

#         Parameters
#         ----------
#         position : Quantity (array-like)
#             The position of the material point (x, y, z).

#         Returns
#         -------
#         bool
#             True if the position satisfies all constraints, False otherwise.
#         """
#         for constraint in self.constraints:
#             if not constraint.contains(position):
#                 return False
#         return True
