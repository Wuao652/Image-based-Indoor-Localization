import numpy as np
def angleDifference(R1, R2):
    temp = np.trace(R1.T @ R2)
    epsilon = 1e-10
    if np.abs(temp - 3) < epsilon:
        angle = 0.0
    else:
        angle = np.arccos((temp -1) / 2) * 360 / np.pi / 2
    return angle
