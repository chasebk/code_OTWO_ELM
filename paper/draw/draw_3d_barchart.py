import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Data generation
alpha = np.linspace(1, 3, 3)
t = np.linspace(0, 6, 6)
T, A = np.meshgrid(t, alpha)


data = np.array(
    [[0.9915, 0.9948, 0.9956, 0.9955, 0.9955, 0.9958],
        [3.0125, 2.9834, 2.9203, 2.9255, 2.9171, 2.9249],
        [16.716, 16.292, 15.378, 15.394, 15.400, 15.363]]
)

# Plotting
fig = plt.figure()
ax = fig.gca(projection = '3d')

Xi = T.flatten()
Yi = A.flatten()
Zi = np.zeros(data.size)

dx = .25 * np.ones(data.size)
dy = .25 * np.ones(data.size)
dz = data.flatten()

ax.set_xlabel('T')
ax.set_ylabel('Alpha')
ax.set_zlabel('Z Label')
ax.bar3d(Xi, Yi, Zi, dx, dy, dz, color = 'w')

plt.show()