from .quantity import Quantity
from .units import definitions as U
from mathematics import Variable, gradients
from .lagrangian import Lagrangian

import numpy as np


class Hamiltonian(object):
    # A class to represent a Hamiltonian function as the Legendre transform of a Lagrangian.

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
        kinetic_energy = potential_energy = Quantity(0, U.J)

        for body in self.pool:
            kinetic_energy += np.dot(body.momentum, body.momentum) / (2 * body.mass)
            potential_energy += body.potential_energy(time)

        return kinetic_energy + potential_energy

        


    def __str__(self):
        return f"Hamiltonian: {self.potentials}"
