import physics.measurements.units as units
from physics.measurements import BaseQuantity, Unit, Prefix, Quantity
import math

length = BaseQuantity(1, 0, 0, 0, 0, 0, 0)
print(length)

metre = Unit(length); 
millimetre = Unit(length, Prefix(1e-3))
metre2 = metre * metre

print(metre)
print(millimetre)

metre_second = Unit(length / BaseQuantity(0, 1, 0, 0, 0, 0, 0), symbol="m/s")
print(metre_second)

print(millimetre * metre_second)
print(metre ** 2)
print(metre_second ** 2)    
print(millimetre ** 2)  
print(millimetre ** 3)  


x = Quantity(1.0, metre)
y = Quantity(2.0, metre)
w = Quantity(2.0, millimetre)
z = x + y
h = x + w
print(x)
print(y)
print(z)
print(h)

print(x * y)
print(x * w)


g = Quantity(9.81, metre_second ** 2)
print(g)

print(g * x)
print(g * y)
# print(x - 3) -> error
# print(x + g) -> error

print(12.0 / x)
print(x ** -1)
print(3.0 * x)
print(x / (2.0 * x))

print(x.to(millimetre))
print(math.floor((3.4459 * x).to(millimetre)))
print(math.ceil((3.4459 * x).to(millimetre)))
