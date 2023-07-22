import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathematics import ParametricEllipticCone
from physics import Quantity
from physics import units as U

# Define the parameters of the elliptic cone.
a = Quantity(1, U.m)
b = Quantity(2, U.m)
height = Quantity(3, U.m)
center = Quantity(np.array([1, 1, 1]), U.m)

# Create a ParametricEllipticCone object.
elliptic_cone = ParametricEllipticCone(a, b, height, center)

# Define the angle parameters of the surface.
u = Quantity(np.linspace(0, 2 * np.pi, 100), U.rad)
v = Quantity(np.linspace(0, 1, 100), U.dimensionless)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(u.value, v.value)

# Calculate the surface points.
x, y, z = elliptic_cone(U, V)

# Plot the surface using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# Set axis labels and plot title.
ax.set_xlabel(f'X [{x.unit}]')
ax.set_ylabel(f'Y [{y.unit}]')
ax.set_zlabel(f'Z [{z.unit}]')
ax.set_title('Surface of an Elliptic Cone')

plt.show()