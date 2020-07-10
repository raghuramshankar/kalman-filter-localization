import math
import numpy as np
from scipy.linalg import sqrtm


def sigma(x, p):
    n = np.shape(x)[0]
    SP = np.zeros((n, 2*n))
    W = np.zeros((1, 2*n))
    for i in range(n):
        SD = sqrtm(p)
        SP[:, i] = (x + (math.sqrt(n) * SD[:, i]).reshape((5, 1))).flatten()
        SP[:, i+n] = (x - (math.sqrt(n) * SD[:, i]).reshape((5, 1))).flatten()
        W[:, i] = 1/(2*n)
        W[:, i+n] = W[:, i]
    return SP.astype(float), W.astype(float)


x = np.ones((5, 1))
p = np.eye(5)

[SP, W] = sigma(x, p)
print(SP)
print(W)

print('end')
