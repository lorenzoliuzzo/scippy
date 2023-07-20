# file    scipp/physics/measurements/basis/definitions.py
# @author  Lorenzo Liuzzo (lorenzoliuzzo@outlook.com)
# @brief   This file contains the definitions of the most common units.
# @date    2023-07-20
# @copyright Copyright (c) 2023

from .. import BaseQuantity


# Define the base quantities
scalar = BaseQuantity(0, 0, 0, 0, 0, 0, 0)
length = BaseQuantity(1, 0, 0, 0, 0, 0, 0)
time = BaseQuantity(0, 1, 0, 0, 0, 0, 0)
mass = BaseQuantity(0, 0, 1, 0, 0, 0, 0)
temperature = BaseQuantity(0, 0, 0, 1, 0, 0, 0)
electric_current = BaseQuantity(0, 0, 0, 0, 1, 0, 0)
substance_amount = BaseQuantity(0, 0, 0, 0, 0, 1, 0)
luminous_intensity = BaseQuantity(0, 0, 0, 0, 0, 0, 1)

# Define the composed base quantities
velocity = length / time
acceleration = velocity / time
force = mass * acceleration
energy = force * length
power = energy / time
pressure = force / length ** 2
electric_charge = electric_current * time
voltage = power / electric_current
capacitance = electric_charge / voltage