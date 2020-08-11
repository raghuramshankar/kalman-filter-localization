# implementation of extended kalman filter using CTRV motion model and linear measurement model

# state matrix:             2D x-y position, heading, velocity and yaw rate (turn rate) (5 x 1)
# input matrix:             --None--
# measurement matrix:       2D x-y position from GPS, velcity and yaw rate (4 x 1)

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd

# initalize global variables
amz = pd.read_csv('AMZ_data_resample_gps.csv')
dt = 0.1                                            # seconds
N = len(amz.lat)  # number of samples


z_noise = np.array([[0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])           # measurement noise


# prior mean
x_0 = np.array([[47.396],                                  # x position    [m]
                [8.6481],                                  # y position    [m]
                [0.0],                                  # yaw           [rad]
                [1.0],                                 # velocity      [m/s]
                [0.1]])                                 # yaw rate      [rad/s]


# prior covariance
p_0 = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]])


# q matrix - process noise
q = np.array([[0.1, 0.0,    0.0,               0.0, 0.0],
              [0.0, 0.1,    0.0,               0.0, 0.0],
              [0.0, 0.0,    np.deg2rad(1.0),   0.0, 0.0],
              [0.0, 0.0,    0.0,               10, 0.0],
              [0.0, 0.0,    0.0,                0.0, np.deg2rad(0.001)]])


# h matrix - measurement model
h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0]])


# r matrix - measurement noise covariance
r = np.array([[0.015, 0.0, 0.0, 0.0],
              [0.0, 0.010, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.0],
              [0.0, 0.0, 0.0, 0.01]])**2


# main program
def main():
    # show_final = int(input('Display final result? (No/Yes = 0/1) : '))
    # show_animation = int(
        # input('Show animation of filter working? (No/Yes = 0/1) : '))
    # if show_animation == 1:
        # show_ellipse = int(
            # input('Display covariance ellipses in animation? (No/Yes = 0/1) : '))
    # else:
        # show_ellipse = 0
    show_final = 1
    show_animation = 0
    show_ellipse = 0
    x_est = x_0
    p_est = p_0
    # x_true = x_0
    # p_true = p_0
    # x_true_cat = np.array([x_0[0, 0], x_0[1, 0]])
    x_est_cat = np.array([x_0[0, 0], x_0[1, 0]])
    z_cat = np.array([x_0[0, 0], x_0[1, 0]])
    for i in range(N):
        # x_true, p_true = extended_prediction(x_true, p_true)
        z = gen_measurement(i)
        if i == (N - 1) and show_final == 1:
            show_final_flag = 1
        else:
            show_final_flag = 0
        postpross(i, x_est, p_est, x_est_cat, z,
                  z_cat, show_animation, show_ellipse, show_final_flag)
        x_est, p_est = extended_kalman_filter(x_est, p_est, z)
        # x_true_cat = np.vstack((x_true_cat, np.transpose(x_true[0:2])))
        z_cat = np.vstack((z_cat, np.transpose(z[0:2])))
        x_est_cat = np.vstack((x_est_cat, np.transpose(x_est[0:2])))
    print('EKF Over')


# generate ground truth measurement vector gz, noisy measurement vector z
def gen_measurement(i):
    # x position [m], y position [m]
    z1 = amz.lat[i]
    z2 = amz.long[i]
    z3 = amz.lin_v_x[i]
    z4 = amz.wZsens[i]
    z = np.array([[z1], [z2], [z3], [z4]])
    return z


# extended kalman filter
def extended_kalman_filter(x_est, p_est, z):
    x_pred, p_pred = extended_prediction(x_est, p_est)
    # return x_pred, p_pred
    x_upd, p_upd = linear_update(x_pred, p_pred, z)
    return x_upd, p_upd


# extended kalman filter nonlinear prediction step
def extended_prediction(x, p):

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
def plot_animation(i, x_est_cat, z):
    if i == 0:
        # plt.plot(x_true_cat[0], x_true_cat[1], '.r')
        plt.plot(x_est_cat[0], x_est_cat[1], '.b')
    else:
        # plt.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r')
        plt.plot(x_est_cat[0:, 0], x_est_cat[0:, 1], 'b')
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


def plot_final(x_est_cat, z_cat):
    fig = plt.figure()
    f = fig.add_subplot(111)
    # f.plot(x_true_cat[0:, 0], x_true_cat[0:, 1], 'r', label='True Position')
    f.plot(x_est_cat[0:, 0], x_est_cat[0:, 1], 'b', label='Estimated Position')
    f.plot(z_cat[0:, 0], z_cat[0:, 1], '+g', label='Noisy Measurements')
    f.set_xlabel('x [m]')
    f.set_ylabel('y [m]')
    f.set_title('Extended Kalman Filter - CTRV Model')
    f.legend(loc='upper left', shadow=True, fontsize='large')
    plt.grid(True)
    plt.show()


def postpross(i, x_est, p_est, x_est_cat, z, z_cat, show_animation, show_ellipse, show_final_flag):
    if show_animation == 1:
        plot_animation(i, x_est_cat, z)
        if show_ellipse == 1:
            plot_ellipse(x_est[0:2], p_est)
    if show_final_flag == 1:
        plot_final(x_est_cat, z_cat)


main()
