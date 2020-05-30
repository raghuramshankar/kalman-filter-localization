import math
import numpy as np
from scipy.sparse.linalg import expm
import scipy.integrate as integrate
from scipy.linalg import sqrtm
import sympy as sp
sp.init_printing(use_latex=True)

x, y, psi, v, dpsi, T = sp.symbols('x y psi v dpsi T')

state = sp.Matrix([x,
                   y,
                   psi,
                   v,
                   dpsi])

test1 = (sp.sin(T * dpsi + psi))
test2 = sp.sin(psi)

F = sp.Matrix([[x + (v/dpsi) * (sp.sin(T * dpsi + psi) - sp.sin(psi))],
               [y + (v/dpsi) * (sp.cos(psi) - sp.cos(T * dpsi + psi))],
               [T * dpsi + psi],
               [v],
               [dpsi]])

jF = F.jacobian(state)

print(F[0], '\n', F[1], '\n', F[2], '\n', F[3], '\n',F[4])

for i in range(0, 5):
    print(jF[i, 0], ' ', jF[i, 1], ' ', jF[i, 2],
          ' ', jF[i, 3], ' ', jF[i, 4], ' ')
