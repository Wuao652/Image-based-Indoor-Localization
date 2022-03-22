############################################
#
# All the math function used in the project
#
############################################
import numpy as np

def angleDifference(R1, R2):
    temp = np.trace(R1.T @ R2)
    epsilon = 1e-10
    if np.abs(temp - 3) < epsilon:
        angle = 0.0
    else:
        angle = np.arccos((temp -1) / 2) * 360 / np.pi / 2
    return angle


def unskew(X):
    # X: 3x3 numpy array
    # so(3) -> R^3
    x = np.array([X(2, 1), X(0, 2), X(1, 0)])
    return x


def wedge(X):
    # X: 4x4 numpy array
    # se(3) -> R^6
    a = unskew(X[0:3, 0:3])
    b = X[0:3, 3]
    return np.concatenate((a, b), axis=None)
