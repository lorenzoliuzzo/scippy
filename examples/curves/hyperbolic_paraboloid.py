import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathematics.curves import Interval, HyperbolicParaboloid
from physics import Quantity
from physics import units as U


# Define the parameters of the hyperbolic paraboloid.
a = Quantity(6, U.m)
b = Quantity(-5, U.m)
center = Quantity(np.array([1, 1, 1]), U.m)
interval_u = Interval(-2, 10)
interval_v = Interval(-2, 10)

# Create a ParametricHyperbolicParaboloid object.
hyperbolic_paraboloid = HyperbolicParaboloid(a, b, center, interval_u, interval_v)

# Define the angle parameters of the surface.
u = Quantity(np.linspace(interval_u.start, interval_u.end, 100), U.m)
v = Quantity(np.linspace(interval_v.start, interval_v.end, 100), U.m)

# Create a meshgrid of the angle parameters.
U, V = np.meshgrid(u.value, v.value)

# Calculate the surface points.
x, y, z = hyperbolic_paraboloid(U, V)

# Plot the surface using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x.value, y.value, z.value, cmap='viridis', alpha=0.8)

# Set axis labels and plot title.
ax.set_xlabel(f'X [{x.unit}]')
ax.set_ylabel(f'Y [{y.unit}]')
ax.set_zlabel(f'Z [{z.unit}]')
ax.set_title('Surface of a Hyperbolic Paraboloid')

plt.show()
