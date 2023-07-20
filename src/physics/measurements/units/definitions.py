# file    scipp/physics/measurements/unit_definitions.py
# @author  Lorenzo Liuzzo (lorenzoliuzzo@outlook.com)
# @brief   This file contains the definitions of the most common units.
# @date    2023-07-20
# @copyright Copyright (c) 2023

from .. import Unit, Prefix, BaseQuantity
from .. import basis


# SI base quantities as units
m = Unit(basis.length)
s = Unit(basis.time)
kg = Unit(basis.mass)
K = Unit(basis.temperature)
A = Unit(basis.electric_current)
mol = Unit(basis.substance_amount)
cd = Unit(basis.luminous_intensity)

# SI derived units with special names and symbols
rad = Unit(basis.scalar, Prefix(1), 'rad')    # Radian
sr = Unit(BaseQuantity(0, 0, 0, 0, 0, 0, 0), Prefix(1), 'sr')      # Steradian
Hz = Unit(basis.time ** -1, Prefix(1), 'Hz')     # Hertz
N = Unit(BaseQuantity(1, -2, 1, 0, 0, 0, 0), Prefix(1), 'N')       # Newton
Pa = Unit(BaseQuantity(-1, -2, 1, 0, 0, 0, 0), Prefix(1), 'Pa')    # Pascal
J = Unit(BaseQuantity(2, -2, 1, 0, 0, 0, 0), Prefix(1), 'J')       # Joule
W = Unit(BaseQuantity(2, -3, 1, 0, 0, 0, 0), Prefix(1), 'W')       # Watt
C = Unit(BaseQuantity(0, 1, 0, 0, 1, 0, 0), Prefix(1), 'C')        # Coulomb
V = Unit(BaseQuantity(2, -3, 1, 0, -1, 0, 0), Prefix(1), 'V')      # Volt
F = Unit(BaseQuantity(-2, 4, -1, 0, 2, 0, 0), Prefix(1), 'F')      # Farad

