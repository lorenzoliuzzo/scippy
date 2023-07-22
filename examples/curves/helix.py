import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mathematics.curves import Helix
from physics import Quantity
from physics import units as U


# Define the parameters of the helix.
radius_helix = Quantity(1, U.m)
pitch_helix = Quantity(3, U.m)
num_turns_helix = 4
center_helix = Quantity(np.array([1, 2, 3]), U.m)
axis_helix = (1, 1, 1)  # Custom axis for the helix orientation
initial_angle_helix = Quantity(np.pi / 4, U.rad)  # Initial angle of the helix

# Create a Helix object.
helix = Helix(radius_helix, pitch_helix, num_turns_helix, center_helix, axis=axis_helix, initial_angle=initial_angle_helix)

# Define the angle parameters of the surface.
polar_angle = Quantity(np.linspace(0, num_turns_helix * 2 * np.pi, 100), U.rad)
azimuthal_angle = Quantity(np.linspace(0, 2 * np.pi, 100), U.rad)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(polar_angle.value, azimuthal_angle.value)

# Calculate the surface points.
x, y, z = helix(U, V)

# Plot the surface using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# Set axis labels and plot title.
ax.set_xlabel(f'X [{x.unit}]')
ax.set_ylabel(f'Y [{y.unit}]')
ax.set_zlabel(f'Z [{z.unit}]')
ax.set_title('Surface of a Helix')

plt.show()