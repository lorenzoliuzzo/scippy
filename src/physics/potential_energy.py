from .quantity import Quantity
from .units import definitions as U


class PotentialEnergy:

    def __init__(self, potential_functions = []):
        """
        Constructor for PotentialEnergy class.

        Parameters
        ----------
        potential_functions : list of callable
            A list of potential functions. Each function should take the same arguments as the __call__ method
            and return the potential energy for the given state.
        """
        self.functions = potential_functions

    def __call__(self, *args, **kwargs):
        """
        Evaluate the potential energy by accumulating values from all potential functions.

        Parameters
        ----------
        *args : tuple
            Positional arguments, which can be in any order and any combination of:
            - Quantity (mass)
            - Quantity (position)
            - Quantity (velocity)
            - Quantity (momentum)
            - Variable (time)

        **kwargs : dict
            Keyword arguments, which can be in any order and any combination of:
            - mass : Quantity
            - position : Quantity (array-like)
            - velocity : Quantity (array-like)
            - momentum : Quantity (array-like)
            - time : Variable

        Returns
        -------
        Quantity
            The accumulated potential energy from all potential functions.
        """
        total_energy = Quantity(0, U.J)
        for potential_func in self.potential_functions:
            total_energy += potential_func(*args, **kwargs)
        return total_energy


    def __str__(self):
        return f"PotentialEnergy: {self.function}"
    
    def __repr__(self):
        return f"PotentialEnergy: {self.function}"
    