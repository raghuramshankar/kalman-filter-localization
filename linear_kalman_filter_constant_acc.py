# implementation of linear kalman filter using 2D constant acceleration model

# measurement matrix:     gps position and imu acceleration in x and y axis (4 x 1)
# state matrix:     position, velocity and acceleration in x and y axis (6 x 1)

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm
import scipy.integrate as integrate
from scipy.linalg import sqrtm

# initalize global variables
dt = 0.01                                            # seconds
N = 50                                            # number of samples
qc = 0.1                                              # process noise magnitude
z_noise = 1                                        # measurement noise magnitude
show_animation = 1
show_ellipse = 0

# initial guesses
# prior mean
x_0 = np.array([[0.0],                              # x position
                [0.0],                              # y position
                [0.0],                              # x velocity
                [0.0],                              # y velocity
                [0.0],                              # x acceleration
                [0.0]])                             # y acceleration

# prior covariance
p_0 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


# motion model
# x_(k) = a_(k-1)*x_(k-1) + q_(k-1)
# a matrix - continuous time motion model
a = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

i_a = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

f_euler = i_a + dt * a
f_exact = expm(dt * a)


gamma = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [0.0],
                  [1.0],
                  [1.0]])

# q matrix - continuous time process noise covariance
q = qc * gamma @ np.transpose(gamma)

q_euler = dt * q
# q_exact =


# measurement model
# y_k = h_k*x_k + r_k

# h matrix - measurement model
h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # x position - GPS
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # y position - GPS
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # x acceleration - IMU
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # y acceleration - IMU

# r matrix - measurement noise covariance
r = np.array([[0.010, 0.0, 0.0, 0.0],           # x position - GPS
              [0.0, 0.015, 0.0, 0.0],           # y position - GPS
              [0.0, 0.0, 0.1, 0.0],             # x acceleration - IMU
              [0.0, 0.0, 0.0, 0.1]])            # y acceleration - IMU
r = r @ r

# main program


def main():
    x = x_0
    p = p_0
    x_true_final_x = x_0[0]
    x_true_final_y = x_0[1]
    x_kalman_final_x = x_0[0]
    x_kalman_final_y = x_0[1]
    z_final_x = x_0[0]
    z_final_y = x_0[1]
    for i in range(N):
        time = dt * i
        z, gz = gen_measurement(i)
        postpross(time, i, x, p, x_true_final_x,
                  x_true_final_y, x_kalman_final_x, x_kalman_final_y, z_final_x, z_final_y, z, gz)
        x, p = kalman_filter(x, p, z)
        x_true_final_x = np.vstack((x_true_final_x, gz[0]))
        x_true_final_y = np.vstack((x_true_final_y, gz[1]))
        z_final_x = np.vstack((z_final_x, z[0]))
        z_final_y = np.vstack((z_final_y, z[1]))
        x_kalman_final_x = np.vstack((x_kalman_final_x, x[0]))
        x_kalman_final_y = np.vstack((x_kalman_final_y, x[1]))
    print('KF Over')


# generate ground truth position gz and noisy position z
def gen_measurement(i):
    gz = np.array([[i],
                   [i],
                   [0.0],
                   [0.0]])
    z = gz + z_noise * np.random.randn(4, 1)
    return z, gz


# linear kalman filter prediction step
def linear_prediction(a, x_hat, p_hat, q):
    x_pred = a @ x_hat
    p_pred = a @ p_hat @ np.transpose(a) + q
    return x_pred, p_pred


# linear kalman filter update step
def linear_update(x_hat, p_hat, y, h, r):
    s = h @ p_hat @ np.transpose(h) + r
    k = p_hat @ np.transpose(h) @ np.linalg.pinv(s)
    v = y - h @ x_hat

    x_upd = x_hat + k @ v
    p_upd = p_hat - k @ s @ np.transpose(k)
    return x_upd, p_upd


# linear kalman filter
def kalman_filter(x, p, z):
    x_pred, p_pred = linear_prediction(f_exact, x, p, q_euler)
    x_upd, p_upd = linear_update(x_pred, p_pred, z, h, r)
    return x_upd, p_upd


def plot_ellipse(x, p):
    phi = np.linspace(0, 2*math.pi, 100)
    p_ellipse = np.array([[p[0, 0], p[0, 1]],
                          [p[1, 0], p[1, 1]]])
    x0 = 3 * sqrtm(p_ellipse)
    xy_1 = np.array([])
    xy_2 = np.array([])
    for i in range(100):
        arr = np.array([[math.sin(phi[i])],
                        [math.cos(phi[i])]])
        arr = x0 @ arr
        xy_1 = np.hstack([xy_1, arr[0]])
        xy_2 = np.hstack([xy_2, arr[1]])
    plt.plot(xy_1 + x[0], xy_2 + x[1], 'r')
    plt.pause(0.00001)


# postprocessing
def postpross(time, i, x, p, x_true_final_x,
              x_true_final_y, x_kalman_final_x, x_kalman_final_y, z_final_x, z_final_y, z, gz):
    if show_animation == 1:
        plt.plot(x[0], x[1], '*b')
        plt.plot(gz[0], gz[1], '*r')
        plt.plot(z[0], z[1], '*k')
#        print(p)
        plt.grid(True)
        plt.pause(0.001)
        if show_ellipse == 1:
            plot_ellipse(x[0:2], p)
    if i == N - 1:
        fig = plt.figure()
        f = fig.add_subplot(111)
        f.plot(x_true_final_x, x_true_final_y, 'r', label='True Position')
        f.plot(x_kalman_final_x, x_kalman_final_y,
               'b', label='Estimated Position')
        f.plot(z_final_x, z_final_y, '*k', label='Noisy Measurements')
        f.set_xlabel('x [m]')
        f.set_ylabel('y [m]')
        f.set_title('Linear Kalman Filter - Constant Acceleration Model')
        f.legend(loc='upper left', shadow=True, fontsize='large')
        plt.grid(True)
        plt.show()


main()
