import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import collections
import os
import pathlib
my_path = pathlib.Path(__file__).parent.absolute()

def error(m, b, points):
    totalError = 0
    for i in range(0, len(points)):
        totalError += (points[i].y - (m * points[i].x + b)) ** 2
    return totalError / float(len(points))

x = y = np.arange(-3.0, 3.0, 0.05)
Point = collections.namedtuple('Point', ['x', 'y'])

m, b = 2, 3
noise = np.random.random(x.size)
points = [Point(xp, m*xp+b+err) for xp,err in zip(x, noise)]

k_history_file = open(os.path.join(my_path, "k_history.txt"), "r")
k_history = list(map(float, k_history_file.read().split(',')))
k_history_file.close()

b_history_file = open(os.path.join(my_path, "b_history.txt"), "r")
b_history = list(map(float, b_history_file.read().split(',')))
b_history_file.close()

for i in range(len(k_history)):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ms = np.linspace(0.0, 4.0, 20)
    bs = np.linspace(1, 5, 20)

    M, B = np.meshgrid(ms, bs)
    zs = np.array([error(mp, bp, points) 
                for mp, bp in zip(np.ravel(M), np.ravel(B))])
    Z = zs.reshape(M.shape)

    ax.plot_surface(M, B, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.8)

    ax.set_xlabel('k')
    ax.set_ylabel('b')
    ax.set_zlabel('MSE')

    ax.plot([k_history[i]], [b_history[i]], [1.], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

    plt.show()