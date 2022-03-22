############################################
#
# All the math function used in the project
#
############################################
import numpy as np
from numpy import sin, cos
from scipy.linalg import expm, logm

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
    y=[[0, -x[2], x[1]], 
       [x[2], 0, -x[0]],
       [-x[1], x[0], 0]]
    return np.array(y)


def getskew(x):
    '''
    The same as skew(x)
    '''

    return skew(x)


def unskew(X):
    '''
    so(3) -> R^3
    Input:
        X: 3x3 numpy array
    '''

    x = np.array([X(2, 1), X(0, 2), X(1, 0)])
    return x


def getinvskew(skew_matrix):
    '''
    The same as unskew(x)
    '''

    x = np.zeros(3)
    x[0] = - skew_matrix[1, 2]
    x[1] = skew_matrix[0, 2]
    x[2] = - skew_matrix[0, 1]
    return x


def wedge(X):
    '''
    se(3) -> R^6
    Input:
        X: 4x4 numpy array
    '''

    a = unskew(X[0:3, 0:3])
    b = X[0:3, 3]
    return np.concatenate((a, b), axis=None)


def hat(x):
    '''
    R^6 -> se(3)
    Input:
        x: 1d numpy array in R^6
    '''

    l1 = np.hstack((skew(x[0:3]), x[3:6]))
    l2 = np.array([0, 0, 0, 0])
    return np.vstack((l1, l2))


def Adjoint_SE3(X):
    '''
    ADJOINT_SE3 Computes the adjoint of SE(3), output a 6x6 matrix
    Input:
        X: 4x4 matrix in SE(3)
    '''

    l1 = np.hstack((X[0:3, 0:3], np.zeros(3, 3)))
    l2 = np.hstack((skew(X[0:3, 3]) @ X[0:3, 0:3], X[0:3, 0:3]))
    return np.vstack((l1, l2))


def LeftJacobian_SO3(w):
    '''
    LEFT JACOBIAN for SO(3), R^(3x3)
    Input:
        w: 1d numpy array with length of 3
    '''

    theta = np.linalg.norm(w)
    A = skew(w)
    if theta == 0:
        return np.eye(3)
    output = np.eye(3) + ((1-cos(theta))/theta**2)*A + ((theta-sin(theta))/theta**3)*(A@A)
    return output


def LeftJacobian_SE3(xi):
    '''
    https://github.com/RossHartley/lie/blob/master/matlab/%2BLie/LeftJacobian_SE3.m
    LEFT JACOBIAN as defined in http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf, R^(6x6)
    Input:
        xi: 1d numpy array with length of 6
    '''

    Phi = xi[0:3]
    phi = np.linalg.norm(Phi)
    Rho = xi[3:6]
    Phi_skew = skew(Phi)
    Rho_skew = skew(Rho)
    J = LeftJacobian_SO3(Phi)

    if (phi == 0):
        Q = 0.5*Rho_skew
    else:
        Q = 0.5*Rho_skew \
            + (phi-sin(phi))/phi**3 * (Phi_skew@Rho_skew + Rho_skew@Phi_skew + Phi_skew@Rho_skew@Phi_skew) \
            - (1-0.5*phi**2-cos(phi))/phi**4 * (Phi_skew@Phi_skew@Rho_skew + Rho_skew@Phi_skew@Phi_skew - 3*Phi_skew@Rho_skew@Phi_skew) \
            - 0.5*((1-0.5*phi**2-cos(phi))/phi**4 - 3*(phi-sin(phi)-(phi**3)/6)/phi**5) \
            * (Phi_skew@Rho_skew@Phi_skew@Phi_skew + Phi_skew@Phi_skew@Rho_skew@Phi_skew)

    l1 = np.hstack((J, np.zeros((3, 3))))
    l2 = np.hstack((Q, J))

    return np.vstack((l1, l2))


def RightJacobianInverse_SE3(xi):
    '''
    Input:
        xi: 1d numpy array with length of 6
    '''

    Jr = RightJacobian_SE3(xi)
    ### TODO: check if this gives the same answer as in the matlab (Jr \ eye(size(Jr))
    output = np.linalg.lstdq(Jr, np.eye(Jr.shape[0]))[0]
    return output


def RightJacobian_SE3(xi):
    '''
    Input:
        xi: 1d numpy array with length of 6
    '''

    output = Adjoint_SE3(expm(hat(-xi))) @ LeftJacobian_SE3(xi)
    return output


def calc_Jr(phi):
    '''
    Input:
        phi: 1x3 matrix
    '''

    if np.linalg.norm(phi) == 0:
        phi = np.ones(3)

    phiv = np.linalg.norm(phi)
    a = phi.reshape(3, 1)/phiv
    Jl = sin(phiv) / phiv * np.eye(3) + (1-sin(phiv)/phiv)*(a@a.T) \
         + (1-cos(phiv))/phiv*getskew(a)
    return Jl, phiv


def calc_Jr(phi):
    '''
    Input:
        phi: 1x3 matrix
    '''

    if np.linalg.norm(phi) == 0:
        phi = np.ones(3)

    phiv = np.linalg.norm(phi)
    a = phi.reshape(3, 1)/phiv
    Jr = sin(phiv) / phiv * np.eye(3) + (1-sin(phiv)/phiv)*(a@a.T) \
         - (1-cos(phiv))/phiv*getskew(a)
    return Jr, phiv


def angleDifference(R1, R2):
    '''
    Computes the angle difference between two rotation matrix R1 and R2
    Input:
        R1: 3x3 rotation matrix
        R2: 3x3 rotation matrix
    '''

    temp = np.trace(R1.T @ R2)
    epsilon = 1e-10
    if np.abs(temp - 3) < epsilon:
        angle = 0.
    else:
        angle = np.arccos((temp -1) / 2) * 360 / np.pi / 2
    return angle


def angleDifference_so3(s):
    '''
    Computes the angle difference of the twist s
    Input:
        s: 1x3 so(3) twist
    '''

    deltaR = expm(skew(s))
    epsilon = 1e-10
    if(abs(trace(deltaR) - 3) < epsilon):
        angle = 0.
    else:
        angle = np.ardcos((np.trace(deltaR) - 1) / 2) * 360 / 2 / np.pi
    return angle


def disDifference(pt1, pt2):
    '''
    Computes the distance difference between two point pt1 and pt2
    Input:
        pt1: 1x3 or 3x1 point
        pt2: 1x3 or 3x1 point
    '''

    pt1 = pt1.reshape(-1)
    pt2 = pt2.reshape(-1)
    tmp = pt1 - pt2
    return np.linalg.norm(tmp)
