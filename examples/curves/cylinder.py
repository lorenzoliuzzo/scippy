import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathematics import ParametricCylinder
from physics import Quantity
from physics import units as U

# Define the parameters of the cylinder.
center = Quantity(np.array([1, 1, 1]), U.m)
radius = Quantity(3, U.m)
height = Quantity(5, U.m)
axis = Quantity(np.array([0, 0, 1]), U.dimensionless)

print("center:", center)
print("radius:", radius)
print("height:", height)
print("axis:", axis)

# Create a ParametricCylinder object.
cylinder = ParametricCylinder(center, radius, height, axis)

# Define the angle parameters of the surface.
u = Quantity(np.linspace(0, 1, 100), U.dimensionless)
v = Quantity(np.linspace(0, 2 * np.pi, 100), U.rad)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(u.value, v.value)

# Calculate the surface points.
x, y, z = cylinder(U, V)

# Plot the surface using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# Set axis labels and plot title.
ax.set_xlabel(f'X [{x.unit}]')
ax.set_ylabel(f'Y [{y.unit}]')
ax.set_zlabel(f'Z [{z.unit}]')
ax.set_title('Surface of a Cylinder')

plt.show()
