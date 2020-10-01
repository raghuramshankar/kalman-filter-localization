# implementation of linear kalman filter using 2D constant acceleration model

# state matrix:             2D x-y position, velocity and acceleration in x and y axis (6 x 1)
# input matrix:             --None--
# measurement matrix:       2D x-y position and acceleration in x and y axis (4 x 1)

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm
from scipy.linalg import sqrtm
import pandas as pd

# initalize global variables
cfs = pd.read_csv('cfs_data_fsn17.csv')
dt = 0.01  # seconds
N = int(len(cfs.XX))-1  # number of samples
# N = 300
qc = 0.0000001  # process noise magnitude

z_noise = 1  # measurement noise magnitude


# prior mean
x_0 = np.array([[0.0],  # x position
                [0.0],  # y position
                [0.0],  # x velocity
                [0.0],  # y velocity
                [0.0],  # x acceleration
                [0.0]  # y acceleration
                ])

# prior covariance
p_0 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


# a matrix - continuous time motion model
a = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


a = expm(dt * a)
gamma = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0]])


# q matrix - continuous time process noise covariance
q = qc * gamma @ np.transpose(gamma)
q_euler = dt * q


# h matrix - measurement model
h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


# r matrix - measurement noise covariance
r = np.array([[0.010, 0.0, 0.0, 0.0],
              [0.0, 0.015, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.0],
              [0.0, 0.0, 0.0, 0.1]])**2


# main program
def main():
    show_final = int(input('Display final result? (No/Yes = 0/1) : '))
    show_animation = int(
        input('Show animation of filter working? (No/Yes = 0/1) : '))
    if show_animation == 1:
        show_ellipse = int(
            input('Display covariance ellipses in animation? (No/Yes = 0/1) : '))
    else:
        show_ellipse = 0
    x_est = x_0
    p_est = p_0
    # x_true_cat = np.array([x_0[0, 0], x_0[1, 0]])
    x_est_cat = np.array([x_0[0, 0], x_0[1, 0]])
    z_cat = np.array([x_0[0, 0], x_0[1, 0]])
    for i in range(N):
        z = gen_measurement(i)
        if i == (N - 1) and show_final == 1:
            show_final_flag = 1
        else:
            show_final_flag = 0
        postpross(x_est, p_est,
                  x_est_cat, z_cat, z, show_animation, show_ellipse, show_final_flag)
        x_est, p_est = kalman_filter(x_est, p_est, z)
        # x_true_cat = np.vstack((x_true_cat, np.transpose(x_true[0:2])))
        z_cat = np.vstack((z_cat, np.transpose(z[0:2])))
        x_est_cat = np.vstack((x_est_cat, np.transpose(x_est[0:2])))
    print('KF Over')


# generate ground truth position x_true and noisy position z
def gen_measurement(i):
    x = float(cfs['XX'][(i)+1])
    y = float(cfs['YY'][(i)+1])
    ax = float(cfs['ax'][(i)+1])
    ay = float(cfs['ay'][(i)+1])
    z = np.array([[x], [y], [ax], [ay]])
    return z


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
    x_pred, p_pred = linear_prediction(a, x, p, q_euler)
    x_upd, p_upd = linear_update(x_pred, p_pred, z, h, r)
    return x_upd, p_upd


# postprocessing
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


def plot_final(x_est_cat, z_cat):
    fig = plt.figure()
    f = fig.add_subplot(111)
    # f.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r', label='True Position')
    f.plot(x_est_cat[0:, 0], x_est_cat[0:, 1], 'b', label='Estimated Position')
    f.plot(z_cat[0:, 0], z_cat[0:, 1], '+g', label='Noisy Measurements')
    f.set_xlabel('x [m]')
    f.set_ylabel('y [m]')
    f.set_title('Linear Kalman Filter - Constant Acceleration Model')
    f.legend(loc='upper right', shadow=True, fontsize='large')
    plt.grid(True)
    plt.show()


def plot_animation(x_est, z):
    # plt.plot(x_true[0], x_true[1], '.r')
    plt.plot(x_est[0], x_est[1], '.b')
    plt.plot(z[0], z[1], '+g')
    plt.grid(True)
    plt.pause(0.001)


def postpross(x_est, p_est, x_est_cat, z_cat, z, show_animation, show_ellipse, show_final_flag):
    if show_animation == 1:
        plot_animation(x_est, z)
        if show_ellipse == 1:
            plot_ellipse(x_est[0:2], p_est)
    if show_final_flag == 1:
        plot_final(x_est_cat, z_cat)


main()
