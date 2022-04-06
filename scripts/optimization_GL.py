# TODO:
# input : K camera intrinsic
#         num_feature
#         orientation
#         position
#         pts3d
#         pts2d

from ast import Param
import numpy as np
from numpy import sin, cos
from scipy.linalg import expm, logm
from numpy.linalg import norm, inv
from utils.cameraParams import generateIntrinsics
# from mathfunc import

def angleDifference_so3(s):
    '''
    Computes the angle difference of the twist s
    Input:
        s: 1x3 so(3) angle
    '''

    deltaR = expm(skew(s))
    epsilon = 1e-10
    if (abs(np.trace(deltaR) - 3) < epsilon):
        angle = 0.
    else:
        angle = np.arccos((np.trace(deltaR) - 1) / 2) * 360 / 2 / np.pi
    return angle


def huber(x, sigma):
    # print("huber")
    output = 0
    if (abs(x) <= sigma):
        output = abs(x)
    else:
        output = np.sqrt(2 * sigma * (np.abs(x) - 0.5 * sigma))
    return output


def CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k, x,
                  Orient, bool_outlier):
    bool_outlier = False
    num_F = 2 * param_num_features  # 562
    F = np.zeros((num_F, 1))  # (562, 1)
    # print("F=", np.shape(F))
    w = np.zeros((num_F, 1))  # (562, 1)
    K = param_k
    # print(num_F)
    # print(F)

    # print(w)
    # print(K)
    # print("hi1")

    robotpose = x[3:6, 0]  # (3, 1)
    robotpose = np.reshape(robotpose, (3, 1))
    R = Orient  # (3, 3)
    # print("robotpose = ", robotpose)
    # print(np.size(robotpose))

    # print("R = ",R)
    # print(np.size(R))
    ind = 0
    # print("hi2")
    # param_num_features = 281
    for j in range(param_num_features - 1):
        # print(observe_pts3D)
        feature = np.reshape(observe_pts3D[j, :], (3, 1))  # (3, 1)

        feature_cam = R @ (feature - robotpose)  # (3, 1)
        # print("R shape = ", np.shape(R))
        # print("feature shape = ", np.shape(feature))
        # print("robotpose shape = ", np.shape(robotpose))
        # print("feature_cam = ", np.shape(feature_cam))

        v = K[0, 0] * feature_cam[0, 0] / feature_cam[2, 0] + K[0, 2]
        u = K[1, 1] * feature_cam[0, 0] / feature_cam[2, 0] + K[1, 2]
        # print(v)
        # print(u)

        # (562, 1)

        u_v = np.zeros((2, 1))
        u_v[0, 0] = u
        u_v[1, 0] = v
        # print(u_v)
        # print(observe_pts2D[j, :])
        # print(np.reshape(observe_pts2D[j, :], (2, -1)))
        F[ind:ind + 2, :] = u_v - np.reshape(observe_pts2D[j, :], (2, -1))
        # print(F)

        w[ind] = np.power((huber(F[ind, 0], 1) / F[ind, 0]), 2)
        w[ind + 1] = np.power((huber(F[ind + 1, 0], 1) / F[ind + 1, 0]), 2)

        ind = ind + 2

    W = np.diag(np.ravel(w))  # 562 * 652
    # print("w=", w)
    # print(np.shape(w))
    # print("W=", W)
    # print(np.shape(W))

    if (bool_outlier == True):
        ind = 0
        delta = 20
        for j in range(param_num_features - 1):
            if (abs(F[ind, :]) > delta or abs(F[ind + 1, :]) > delta):
                F[ind:ind + 2, :] = 0

            ind = ind + 2

    return F, W


def skew(x):
    '''
    skew function, R^3 -> so(3)
    Input:
        x: 1x3 or 3x1 numpy array
    '''

    x = x.reshape(-1)
    if len(x) != 3:
        print('dimension wrong, return np.zeros((3, 3))')
        return np.zeros((3, 3))
    y = [[0, -x[2], x[1]],
         [x[2], 0, -x[0]],
         [-x[1], x[0], 0]]
    return np.array(y)


def JacobianF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k, x,
                 Orient):
    num_F = 2 * param_num_features
    J = np.zeros((num_F, 6))
    K = param_k  # (3, 3)
    robotpose = x[3:6, 0]  # (3, 1)
    R = Orient  # (3, 3)
    ind = 0

    robotpose = x[3:6, 0]  # (3, 1)
    robotpose = np.reshape(robotpose, (3, 1))
    R = Orient  # (3, 3)
    ind = 0

    for j in range(param_num_features - 1):
        # print(observe_pts3D)
        feature = np.reshape(observe_pts3D[j, :], (3, 1))  # (3, 1)

        feature_cam = R @ (feature - robotpose)  # (3, 1)

        v = K[0, 0] * feature_cam[0, 0] / feature_cam[2, 0] + K[0, 2]
        u = K[1, 1] * feature_cam[0, 0] / feature_cam[2, 0] + K[1, 2]

        PI_feature_cam = [
            [0, K[1, 1] / feature_cam[2, 0], -K[1, 1] * feature_cam[1, 0] / np.power(feature_cam[2, 0], 2)],
            [K[0, 0] / feature_cam[2, 0], 0, -K[0, 0] * feature_cam[1, 0] / np.power(feature_cam[2, 0], 2)]]

        J[ind:ind + 2, 0:3] = PI_feature_cam @ (-R @ skew(feature - robotpose))

        J[ind:ind + 2, 3:6] = PI_feature_cam @ (-R)

        ind = ind + 2

    print(np.shape(J))
    return J


def optimizationLS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k):
    x0 = np.zeros((6, 1))

    if (1):
        Orient = observe_orientation.T
        x0[3:6, 0] = observe_robotPose

    print(x0)
    print(Orient)

    F, P = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features,
                         param_k, x0, Orient, False)

    J = JacobianF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k,
                     x0, Orient)

    reproerror1 = 0
    # param_num_features = 281
    for i in range(param_num_features - 1):
        # F = np.zeros((num_F, 1))  # (562, 1)
        reproerror1 = reproerror1 + np.sqrt(F[2 * i, 0] ** 2 + F[2 * i + 1, 0] ** 2)
    reproerror1 = reproerror1 / param_num_features

    # print(reproerror1)

    # ============================  No outlier =========================#
    M = 0.000001
    min_FX_old = 10000000000000
    k = 0
    x = x0;
    # print("J shape", np.shape(J[1]))
    # J_dim_2 = np.shape(J[1])
    # J_dim = J_dim_2[0]
    J_dim = J.shape[1]
    # print("J_dim", J_dim)
    I = np.identity(J_dim)
    # I = np.ones((J_dim, 1))
    # print("I = ",I)

    while ((np.conj(F).T @ P @ F)[0, 0] > M and abs(np.conj(F).T @ P @ F - min_FX_old)[0, 0] > M and k < 50):

        min_FX_old = np.conj(F).T * P * F

        u = 0.0000001
        print(np.shape(min_FX_old))

        first = (np.conj(J).T @ P @ F)
        # print("first ", np.shape(first))
        second = -(np.conj(J).T @ P @ J + u * I)
        # print("second ", np.shape(second))
        d = -inv(second) @ first  # (6, 1)
        # print("d = ", d)
        # print("d =", np.shape(d))
        x_new = x + d
        # print("x_new", x_new)
        Orient_new = Orient * expm(skew(d[0:3, 0]))
        # print("Orient", np.shape(Orient_new))

        F_new, P_new = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D,
                                     param_num_features, param_k, x, Orient, False)
        min_FX = np.conj(F_new).T @ P_new @ F_new
        # print("min_FX", min_FX)
        t = 1
        while (min_FX[0, 0] > min_FX_old[0, 0] and t < 6):

            d = -inv(np.conj(J).T @ P_new @ J + u * I) @ (np.conj(J).T @ P_new @ F)
            x_new = x + d
            Orient_new = Orient * expm(skew(d[0:3, 0]))

            F_new, P_new = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D,
                                         param_num_features, param_k, x, Orient, 1)
            min_FX = np.conj(F_new).T @ P_new @ F_new
            u = u * 100
            t = t + 1

            if (t == 6):
                break

            Orient = Orient * expm(skew(d[0:3, 0]))
            x = x + d

            k = k + 1

            if (k > 5):
                F, P = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D,
                                     param_num_features, param_k, x, Orient, 1)
            else:
                F, P = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D,
                                     param_num_features, param_k, x, Orient, False)

            J = JacobianF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features,
                             param_k, x, Orient)

            # print("end")

        var_1_1 = (np.conj(F).T @ P @ F)
        print("var_1_1", var_1_1)
        var_1_2 = (2 * param_num_features)
        print("var_1_2", var_1_2)

        var_1 = (var_1_1 / var_1_2) * np.ones((6, 6))
        print("var_1", var_1)
        var_2 = inv(np.conj(J).T @ P @ J)
        print("var_2", var_2)

        var_res = var_1 @ var_2
        print("var_res", var_res)

        # var = ((np.conj(F).T @ P @ F) / (2 * param_num_features)) @ inv(np.conj(J).T @ P @ J)
        var = var_res
        print(np.shape(var))
        x_std = np.sqrt(np.diag(var))  # (6, 1)
        print("x_std", x_std)
        print("X_std_size", np.shape(x_std))
        angle_std = angleDifference_so3(x_std[0:3, 0])
        angle_var = np.power(angle_std, 2)
        position_var = np.power(norm(x_std[3:6]), 2)
        robotpose = np.conj(x[3:6]).T
        F = CalculateF_LS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features,
                          param_k, x, Orient, 0)
        reproerror2 = 0
        for i in range(param_num_features - 1):
            reproerror2 = reproerror2 + np.sqrt(np.power(F[2 * i - 1], 2) + np.power(F[2 * i], 2))

        reproerror2 = reproerror2 / param_num_features
        print("reproerror2", reproerror2)
        # print('Average reprojection error: ', num2str(reproerror1),' --->', num2str(reproerror2))
        # Orient = np.conj(Orient).T

    return F, P, J

    # # return Orient #, robotpose, reproerror2, angle_var, position_var


# optimizationLS(observe_orientation, observe_robotPose, observe_pts2D, observe_pts3D, param_num_features, param_k)
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
