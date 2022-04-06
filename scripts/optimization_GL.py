# TODO:
# input : K camera intrinsic
#         num_feature
#         orientation
#         position
#         pts3d
#         pts2d

from ast import Param
import numpy as np
from scipy.linalg import expm
from numpy.linalg import norm, inv
from utils.cameraParams import generateIntrinsics
from utils.mathfunc import *


def huber(x, sigma):
    output = 0
    if (abs(x) <= sigma):
        output = abs(x)
    else:
        output = np.sqrt(2 * sigma * (np.abs(x) - 0.5 * sigma))
    return output


def CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, 
                  x, Orient, bool_outlier=False):

    num_F = 2 * param_num_features
    F = np.zeros((num_F, 1))
    w = np.zeros((num_F, 1))
    K = param_k

    robotpose = x[3:6, :]  # (3, 1)
    R = Orient  # (3, 3)

    for j in range(param_num_features):
        feature = np.reshape(observe_pts3D[j, :], (3, 1))  # (3, 1)
        feature_cam = R @ (feature - robotpose)  # (3, 1)

        v = K[0, 0] * feature_cam[0, 0] / feature_cam[2, 0] + K[0, 2]
        u = K[1, 1] * feature_cam[1, 0] / feature_cam[2, 0] + K[1, 2]

        u_v = np.zeros((2, 1))
        u_v[0, 0] = u
        u_v[1, 0] = v

        ind = 2*j
        F[ind:ind + 2, :] = u_v - np.reshape(observe_pts2D[j, :], (2, -1))
        w[ind] = np.power((huber(F[ind, 0], 1) / F[ind, 0]), 2)
        w[ind + 1] = np.power((huber(F[ind + 1, 0], 1) / F[ind + 1, 0]), 2)

    W = np.diag(np.ravel(w))

    if (bool_outlier):
        delta = 20
        for j in range(param_num_features):
            ind = 2*j
            if (abs(F[ind, :]) > delta or abs(F[ind + 1, :]) > delta):
                F[ind:ind + 2, :] = 0

    return F, W


def JacobianF_LS(observe_pts3D, param_num_features, param_k, x, Orient):
    num_F = 2 * param_num_features
    J = np.zeros((num_F, 6))
    K = param_k  # (3, 3)

    robotpose = x[3:6, :]  # (3, 1)
    R = Orient  # (3, 3)

    for j in range(param_num_features):
        feature = np.reshape(observe_pts3D[j, :], (3, 1))  # (3, 1)
        feature_cam = R @ (feature - robotpose)  # (3, 1)

        PI_feature_cam = [[0, K[1, 1] / feature_cam[2, 0], -K[1, 1] * feature_cam[1, 0] / np.power(feature_cam[2, 0], 2)],
                          [K[0, 0] / feature_cam[2, 0], 0, -K[0, 0] * feature_cam[0, 0] / np.power(feature_cam[2, 0], 2)]]

        ind = 2*j
        J[ind:ind + 2, 0:3] = PI_feature_cam @ (-R @ skew(feature - robotpose))
        J[ind:ind + 2, 3:6] = PI_feature_cam @ (-R)

    return J


def optimizationLS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k):
    x0 = np.zeros((6, 1))

    if (1):
        Orient = observe_orientation.T
        x0[3:6, 0] = observe_robotPose

    F, P = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x0, Orient)

    J = JacobianF_LS(observe_pts3D, param_num_features, param_k, x0, Orient)

    reproerror1 = 0
    for i in range(param_num_features):
        reproerror1 = reproerror1 + np.sqrt(F[2 * i, 0] ** 2 + F[2 * i + 1, 0] ** 2)
    reproerror1 = reproerror1 / param_num_features

    # ============================  No outlier =========================#
    M = 1E-6
    min_FX_old = 1E13
    k = 0
    x = x0
    J_dim = J.shape[1] # should be 6
    I = np.identity(J_dim)

    while ((np.conj(F).T @ P @ F).item() > M and abs(np.conj(F).T @ P @ F - min_FX_old).item() > M and k < 50):

        min_FX_old = np.conj(F).T @ P @ F

        u = 1E-7

        # solve the linear equation
        first = (np.conj(J).T @ P @ F)
        second = -(np.conj(J).T @ P @ J + u * I)
        d = inv(second) @ first  # (6, 1)

        x_new = x + d
        Orient_new = Orient @ expm(skew(d[0:3, 0]))

        F_new, P_new = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x_new, Orient_new)
        min_FX = np.conj(F_new).T @ P_new @ F_new
        t = 1
        while (min_FX.item() > min_FX_old.item() and t < 6):

            d = -inv(np.conj(J).T @ P_new @ J + u * I) @ (np.conj(J).T @ P_new @ F)
            x_new = x + d
            Orient_new = Orient @ expm(skew(d[0:3, 0]))

            F_new, P_new = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x_new, Orient_new)
            min_FX = np.conj(F_new).T @ P_new @ F_new
            u *= 100
            t += 1

        if (t == 6):
            break

        Orient = Orient @ expm(skew(d[0:3, 0]))
        x = x + d

        k += 1

        if (k > 5):
            F, P = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x, Orient) # No outlier rejection
        else:
            F, P = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x, Orient)

        J = JacobianF_LS(observe_pts3D, param_num_features, param_k, x, Orient)

    # var = ((np.conj(F).T @ P @ F) / (2 * param_num_features)) @ inv(np.conj(J).T @ P @ J)
    var_1_1 = np.conj(F).T @ P @ F
    var_1_2 = 2 * param_num_features
    var_1 = var_1_1 / var_1_2   # * np.ones((6, 6))
    var_2 = inv(np.conj(J).T @ P @ J)
    var_res = var_1 * var_2
    var = var_res
    # print("var_res\n", var_res)

    x_std = np.sqrt(np.diag(var))  # (6,)
    angle_std = angleDifference_so3(x_std[0:3])
    angle_var = np.power(angle_std, 2)
    position_var = np.power(norm(x_std[3:6]), 2)
    robotpose = np.conj(x[3:6]).reshape(3) #

    F, _ = CalculateF_LS(observe_pts2D, observe_pts3D, param_num_features, param_k, x, Orient)
    reproerror2 = 0
    for i in range(param_num_features):
        reproerror2 = reproerror2 + np.sqrt(np.power(F[2*i], 2) + np.power(F[2*i+1], 2))

    reproerror2 = reproerror2 / param_num_features
    print(f'Average reprojection error: {reproerror1.item():.4f} ---> {reproerror2.item():.4f}')

    Orient = np.conj(Orient).T

    return Orient, robotpose, reproerror2, angle_var, position_var


if __name__ == "__main__":
    print(f'hello world from optimization')

    # obsevation
    observe_pts2D = np.loadtxt('../data/observe.pts2D.txt')
    observe_pts3D = np.loadtxt('../data/observe.pts3D.txt')
    observe_orientation = np.array([[0.451706456386172, 0.571960139035106, -0.684706416366890],
                                    [0.891649603806108, -0.263293216519696, 0.368290192873940],
                                    [0.0303687551845320, -0.776877262821905, -0.628919277188167]])
    observe_robotPose = np.array([1.34420000000000, 0.268600000000000, 1.72490000000000])

    # params
    param_num_features = 190
    cameraParams = generateIntrinsics()
    param_k = np.array(cameraParams['IntrinsicMatrix']).T
    optimizationLS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k)
