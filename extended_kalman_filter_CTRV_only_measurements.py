# implementation of linear kalman filter using CTRV model

# state matrix:                     2D x-y position, heading, velocity and yaw rate (turn rate) (5 x 1)
# input matrix:                     --None--
# measurement matrix:               2D x-y position from GPS (2 x 1)

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm
import scipy.integrate as integrate
from scipy.linalg import sqrtm

# initalize global variables
dt = 0.1                                            # seconds
N = 63                                             # number of samples
u_noise = np.array([[0.1, 0.0],
                    [0.0, np.deg2rad(10)]])         # input noise
z_noise = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])           # measurement noise


# prior mean
x_0 = np.array([[0.0],                                  # x position    [m]
                [0.0],                                  # y position    [m]
                [0.0],                                  # yaw           [rad]
                [10.0],                                 # velocity      [m/s]
                [1.0]])                                 # yaw rate      [rad/s]


# prior covariance
p_0 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]])


# q matrix - process noise
q = np.array([[10.0, 0.0,    0.0,               0.0, 0.0],
              [0.0, 10.0,    0.0,               0.0, 0.0],
              [0.0, 0.0,    np.deg2rad(10.0),   0.0, 0.0],
              [0.0, 0.0,    0.0,               10.0, 0.0],
              [0.0, 0.0,    0.0,                0.0, np.deg2rad(1.0)]])*0

# h matrix - measurement model
h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0]])

# r matrix - measurement noise covariance
r = np.array([[0.015, 0.0, 0.0, 0.0],
              [0.0, 0.010, 0.0, 0.0],
              [0.0, 0.0, 0.010, 0.0],
              [0.0, 0.0, 0.0, 0.015]])**2


# main program
def main():
    show_final = 0
    show_animation = 1
    show_ellipse = 0
    x_est = x_0
    p_est = p_0
    x_true = x_0
    p_true = p_0
    x_true_cat = np.array([x_0[0, 0], x_0[1, 0]])
    x_est_cat = np.array([x_0[0, 0], x_0[1, 0]])
    z_cat = np.array([x_0[0, 0], x_0[1, 0]])
    for i in range(N):
        # u, gu = gen_input()
        x_true, p_true = nonlinear_prediction(x_true, p_true)
        z, gz = gen_measurement(x_true)
        if i == (N - 1):
            show_final = 1
        postpross(x_true, x_true_cat, x_est, p_est, x_est_cat, z,
                  z_cat, show_animation, show_ellipse, show_final)
        x_est, p_est = extended_kalman_filter(x_est, p_est, z)
        x_true_cat = np.vstack((x_true_cat, np.transpose(x_true[0:2])))
        z_cat = np.vstack((z_cat, np.transpose(z[0:2])))
        x_est_cat = np.vstack((x_est_cat, np.transpose(x_est[0:2])))
    print('EKF Over')


# generate ground truth measurement vector gz, noisy measurement vector z
def gen_measurement(x_true):
    # x position [m], y position [m]
    gz = h @ x_true
    z = gz + z_noise @ np.random.randn(4, 1)
    return z, gz


# generate ground truth input vector gu, noisy input vector u
def gen_input():
    # velocity [m/s], yaw rate [rad/s]
    gu = np.array([[1.0], [0.1]])
    u = gu + u_noise @ np.random.randn(2, 1)
    return u, gu


# extended kalman filter
def extended_kalman_filter(x_est, p_est, z):
    x_pred, p_pred = nonlinear_prediction(x_est, p_est)
    # return x_pred, p_pred
    x_upd, p_upd = linear_update(x_pred, p_pred, z)
    return x_upd, p_upd


# extended kalman filter nonlinear prediction step
def nonlinear_prediction(x, p):

    # f(x)
    x[0] = x[0] + (x[3]/x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
    x[1] = x[1] + (x[3]/x[4]) * (- np.cos(x[4] * dt + x[2]) + np.cos(x[2]))
    x[2] = x[2] + x[4] * dt
    x[3] = x[3]
    x[4] = x[4]

    a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
    a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) -
                (x[3]/x[4]**2)*(np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) -
                (x[3]/x[4]**2)*(-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    jF = np.matrix([[1.0, 0.0, a13, a14, a15],
                    [0.0, 1.0, a23, a24, a25],
                    [0.0, 0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])
    x_pred = x
    p_pred = jF @ p @ np.transpose(jF) + q
    return x_pred.astype(float), p_pred.astype(float)


# extended kalman filter linear update step
def linear_update(x_pred, p_pred, z):
    s = h @ p_pred @ np.transpose(h) + r
    k = p_pred @ np.transpose(h) @ np.linalg.pinv(s)
    v = z - h @ x_pred

    x_upd = x_pred + k @ v
    p_upd = p_pred - k @ s @ np.transpose(k)
    return x_upd.astype(float), p_upd.astype(float)


# postprocessing
def plot_animation(x_true, x_est, z):
    plt.plot(x_true[0], x_true[1], '.r')
    plt.plot(x_est[0], x_est[1], '.b')
    plt.plot(z[0], z[1], '+g')
    plt.grid(True)
    plt.pause(0.001)


def plot_ellipse(x_est, p_est):
    phi = np.linspace(0, 2 * math.pi, 100)
    p_ellipse = np.array(
        [[p_est[0, 0], p_est[0, 1]], [p_est[1, 0], p_est[1, 1]]])
    x0 = 3 * sqrtm(p_ellipse)
    xy_1 = np.array([])
    xy_2 = np.array([])
    for i in range(100):
        arr = np.array([[math.sin(phi[i])], [math.cos(phi[i])]])
        arr = x0 @ arr
        xy_1 = np.hstack([xy_1, arr[0]])
        xy_2 = np.hstack([xy_2, arr[1]])
    plt.plot(xy_1 + x_est[0], xy_2 + x_est[1], 'r')
    plt.pause(0.00001)


def plot_final(x_true_cat, x_est_cat, z_cat):
    fig = plt.figure()
    f = fig.add_subplot(111)
    f.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r', label='True Position')
    f.plot(x_est_cat[0:, 0], x_est_cat[0:, 1], 'b', label='Estimated Position')
    f.plot(z_cat[0:, 0], z_cat[0:, 1], '+g', label='Noisy Measurements')
    f.set_xlabel('x [m]')
    f.set_ylabel('y [m]')
    f.set_title('Extended Kalman Filter - CTRV Model')
    f.legend(loc='upper left', shadow=True, fontsize='large')
    plt.grid(True)
    plt.show()


def postpross(x_true, x_true_cat, x_est, p_est, x_est_cat, z, z_cat, show_animation, show_ellipse, show_final):
    if show_animation == 1:
        plot_animation(x_true, x_est, z)
        if show_ellipse == 1:
            plot_ellipse(x_est[0:2], p_est)
    if show_final == 1:
        plot_final(x_true_cat, x_est_cat, z_cat)


main()
