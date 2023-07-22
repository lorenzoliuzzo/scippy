import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathematics.curves import Cone
from physics import Quantity
from physics import units as U

# Define the parameters of the cone.
apex = Quantity(np.array([0, 0, 0]), U.m)
base_radius = Quantity(1, U.m)
height = Quantity(3, U.m)

# Create a Cone object.
cone = Cone(apex, base_radius, height)

# Define the parameters of the surface.
u_values = Quantity(np.linspace(0, 1, 100), U.dimensionless)
v_values = Quantity(np.linspace(0, 2 * np.pi, 100), U.rad)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(u_values.value, v_values.value)

print("U:", U)
print("V:", V)

# Calculate the surface points.
x, y, z = cone(U, V)

# # Plot the surface using matplotlib.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# # Set axis labels and plot title.
# ax.set_xlabel(f'X [{x.unit}]')
# ax.set_ylabel(f'Y [{y.unit}]')
# ax.set_zlabel(f'Z [{z.unit}]')
# ax.set_title('Surface of a Cone')

# plt.show()
