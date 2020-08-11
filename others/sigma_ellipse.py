import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

mu = np.array([[-2.0],
               [1.0]])
P = np.array([[4.0, -2.0],
              [-2.0, 2.0]])


def sigmaEllipse2D(mu, sigma):
    phi = np.linspace(0, 2*math.pi, 100)
    x0 = 3 * sqrtm(sigma)
    xy_1 = np.array([])
    xy_2 = np.array([])
    for i in range(100):
        arr = np.array([[math.sin(phi[i])],
                        [math.cos(phi[i])]])
        arr = mu + x0 @ arr
        xy_1 = np.hstack([xy_1, arr[0]])
        xy_2 = np.hstack([xy_2, arr[1]])
    plt.plot(mu[0], mu[1], '*b')
    plt.plot(xy_1, xy_2, 'r')
    plt.grid(True)
    plt.show()


sigmaEllipse2D(mu, P)
