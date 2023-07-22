from physics.potential_energy import PotentialEnergy
from physics.quantity import Quantity

import numpy as np


class Elastic(PotentialEnergy):
    def __init__(self, spring_constant, rest_length):
        self.spring_constant = spring_constant
        self.rest_length = rest_length

    def __call__(self, position, mass = None, velocity = None, momentum = None, time = None):      
        displacement = np.linalg.norm(position) - self.rest_length
        return 0.5 * self.spring_constant * displacement ** 2

class GravitationalPotential:
    def __init__(self, gravitational_constant, reference_position):
        self.gravitational_constant = gravitational_constant
        self.reference_position = reference_position

    def __call__(self, mass, position):
        displacement = position - self.reference_position
        return -self.gravitational_constant * mass / np.linalg.norm(displacement)
