from .quantity import Quantity
from .units import definitions as U
from mathematics import Variable

import numpy as np


class Lagrangian(object):
    # A class to represent a Lagrangian function.

    def __init__(self, bodies=None):
        """
        Parameters
        ----------
        bodies : list of MaterialPoint, optional
            A list of material points (bodies) that will be part of the Lagrangian. (Default is None)
        """
        if bodies is None:
            self.pool = []
        else:
            self.pool = [bodies]


    def add_body(self, body):
        """
        Add a material point (body) to the Lagrangian.

        Parameters
        ----------
        body : MaterialPoint
            The material point to be added.
        """
        self.bodies.append(body)


    def __call__(self, time = Variable('t', Quantity(0, U.s))):
        """
        Parameters
        ----------
        mp      The material point.
        time    The time of the evaluation.
        """

        # for body in self.pool:
        #     body.update(time)

        # Evaluate the kinetic energy and the potential energy.
        kinetic_energy = potential_energy = Quantity(0, U.J)

        for body in self.pool:
            kinetic_energy += 0.5 * body.mass * np.dot(body.velocity, body.velocity)
            potential_energy += body.potential_energy(time)

        return kinetic_energy - potential_energy
    
    
    def __str__(self):
        return f"Lagrangian: {self.pool}"
