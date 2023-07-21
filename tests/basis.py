# file: ./basis.py 
from physics import BaseQuantity
from physics import basis as basis

# Creating a base quantity
length = BaseQuantity(1, 0, 0, 0, 0, 0, 0)
print(length)

time = BaseQuantity(0, 1, 0, 0, 0, 0, 0)
print(time)

# Creating a composed base quantity
velocity = length / time
print(velocity)

# Using the predefined base quantities from basis
print(basis.length)
print(basis.time)
print(basis.velocity)
print(basis.force)