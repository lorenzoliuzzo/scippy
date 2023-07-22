from physics import Quantity, MaterialPoint
from physics import units as U
from mathematics import Variable, gradients
from mathematics.curves import Sphere

import numpy as np

mp = MaterialPoint(Quantity(1, U.kg), Quantity(np.array([1, 0, 0]), U.m), Quantity(np.array([4, 5, 6]), U.m / U.s))
print(mp)

print(mp.momentum())
print(mp.kinetic_energy())

grad = gradients(mp.momentum(), [mp.velocity])
print(grad[mp.velocity])

grad = gradients(mp.kinetic_energy(), [mp.velocity])
print(grad[mp.velocity])


sphere = Sphere(Quantity(1, U.m), Quantity(np.array([0, 0, 0]), U.m))

u_var = Variable('theta', Quantity(0, U.rad))
v_var = Variable('phi', Quantity(0, U.rad))

print(u_var)
print(v_var)

x, y, z = sphere(u_var, v_var)
print(x)
print(y)
print(z)

dx = gradients(x, [u_var, v_var])
print('dxdtheta ->', dx[u_var])
print('dxdphi ->', dx[v_var])

dy = gradients(y, [u_var, v_var])
print('dydtheta ->', dy[u_var])
print('dydphi ->', dy[v_var])

dz = gradients(z, [u_var, v_var])
print('dzdtheta ->', dz[u_var])
print('dzdphi ->', dz[v_var])

# mp.constrain_to_surface(sphere)