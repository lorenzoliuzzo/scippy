from physics import Quantity, Unit, PotentialEnergy
from physics import basis as bs
from physics import units as U
from mathematics import Variable
# from mathematics.curves import ParametricSurface

import numpy as np


class MaterialPoint:
    """
    A material point in space.
    """

    def __init__(self, name, x, xdot, mass, potentials : PotentialEnergy = []):
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

        if x.unit == Unit(bs.length):
            self.position = Variable('x', x)
        else:
            raise ValueError("Invalid unit for position.")

        if xdot.unit == Unit(bs.velocity):
            self.velocity = Variable('xdot', xdot)
            self.momentum = Variable('p', mass * self.velocity)

        elif xdot.unit == Unit(bs.momentum):
            self.momentum = Variable('p', xdot)
            self.velocity = Variable('xdot', self.momentum / mass)

        else:
            raise ValueError("Invalid unit for velocity.")
        
        if mass.unit == Unit(bs.mass):
            self.mass = mass
        else:
            raise ValueError("Invalid unit for mass.")

        self.name = name
        self.potentials = potentials


    def __str__(self):
        return f"{self.name}: \nmass: {self.mass}, \nposition: {self.position}, \nvelocity: {self.velocity}, \nmomentum: {self.momentum}, \npotential energy: {self.potential_energy()}"
    

    def potential_energy(self, time = Variable('t', Quantity(0, U.s))):
        """
        Evaluate the potential energy of the material point.
        """
        potential_energy = Quantity(0, U.J)
        for potential in self.potentials:
            potential_energy += potential(self.position, self.velocity, self.momentum, self.mass, time)
        return potential_energy
    

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
