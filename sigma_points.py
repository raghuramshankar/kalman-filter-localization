import math
# import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm
import scipy.integrate as integrate
from scipy.linalg import sqrtm

def sigma(x, p):
    n = np.shape(x)[0]
    SP = np.zeros((n, 2*n))
    W = np.zeros((1, 2*n))
    for i in range(n - 1):
        SD = sqrtm(p)
        SP[:,i] = (x + math.sqrt(n) * SD[:,i])[:,i]
        SP[:,i+n] = (x - math.sqrt(n) * SD[:,i])[:,i]
        W[:,i] = 1/(2*n)
        W[:,i+n] = W[:,i]    
    return SP, W

# x = np.array([  [0.0],                                  # x position    [m]
#                 [0.0],                                  # y position    [m]
#                 [0.0],                                  # velocity      [m/s]
#                 [0.0],                                  # yaw           [rad]
#                 [0.0]   ])                              # yaw rate      [rad/s]
# p = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 1.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 1.0]])

x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
p = np.eye(5)

[SP, W] = sigma(x, p)
print(SP)
print(W)