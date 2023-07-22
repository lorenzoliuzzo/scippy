import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathematics.curves import Torus
from physics import Quantity
from physics import units as U

# Define the parameters of the torus.
radius = Quantity(3, U.m)
inner_radius = Quantity(1, U.m)
centre = Quantity(np.array([1, 1, 1]), U.m)

print("radius:", radius)
print("inner_radius:", inner_radius)
print("centre:", centre)

# Create a ParametricTorus object.
torus = Torus(radius, inner_radius, centre)

# Define the angle parameters of the surface.
polar_angle = Quantity(np.linspace(0, 2 * np.pi, 100), U.rad)
azimuthal_angle = Quantity(np.linspace(0, 2. * np.pi, 100), U.rad)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(polar_angle.value, azimuthal_angle.value)

# Calculate the surface points.
x, y, z = torus(U, V)

# Plot the surface using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# Set axis labels and plot title.
ax.set_xlabel(f'X [{x.unit}]')
ax.set_ylabel(f'Y [{y.unit}]')
ax.set_zlabel(f'Z [{z.unit}]')
ax.set_title('Surface of a Torus')

plt.show()