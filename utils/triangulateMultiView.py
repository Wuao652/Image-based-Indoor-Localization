### Rewrite the triangulateMultiview.m in matlab
import numpy as np
import scipy

def cameraMatrix(camParams, Rotation, translation):
    '''
    camParams:    the parameters of the camera
    Rotation:     Roation of the current camera
    translation:  translation of the current camera
    '''
    K = camParams['IntrinsicMatrix']
    R = Rotation.reshape((3, 3))
    t = translation.reshape((1, 3))
    P = np.vstack((R, t))@K
    return P



def triangulateOnePoint(track, cameraMatrices):
    viewIds = np.array([e[0] for e in track])
    points  = np.array([e[1] for e in track])
    numViews = len(viewIds)
    A = np.zeros((numViews*2, 4))

    for i in range(numViews):
        # Check if the viewId exists
        try:
            viewIds[i] in cameraMatrices
        except KeyError:
            print('The viewId {} does not exist in the cameraMatrices'.format(viewIds[i]))
            print('Please check that'.format(viewIds[i]))
            exit(0)

        P = cameraMatrices[viewIds[i]].T
        idx = 2*i
        A[idx:idx+2, :] = points[i].reshape(-1, 1) * P[2, :] - P[0:2, :]

    _, _, V = np.linalg.svd(A)
    X = V[:, -1]
    X = X/X[-1]

    return X[0:3]



def reprojectionErrors(points3d, cameraMatrices, tracks):
    pass


def reprojectPoint(p3dh, viewIds, cameraMatrices):
    pass


def triangulateMultiView(pointTracks, camPoses, camParams):
    ''' 
    Input:
    pointTracks should be a dictionary with,
        Key:       Index of the keypoint in the first image (the test image)
        Value:     A list of (ViewId, keypoint) passing the match of 
                   the corresponding keypoint in the test image
    camPoses       A list of dictionary containing: 'ViewId', 'Orientation',
                   and 'Location'.
    camParams      Camera parameters

    Output is also a dictionary with,
        Key:       index of the keypoint in the first image (the test image)
        Value:     A 3D points in the world frame
    '''
    numTracks = len(pointTracks)
    points3d  = np.zeros((numTracks, 3))

    numCameras = len(camPoses)
    cameraMatrices = dict()

    # for i, _ in enumerate(pointTracks):
    for i in range(numCameras):
        id = camPoses[i]['ViewId']
        R  = camPoses[i]['Orientation']
        t  = camPoses[i]['Location']
        cameraMatrices[id] = cameraMatrix(camParams, R.T, -t@R.T)

    for i, key in enumerate(pointTracks):
        track = pointTracks[key]
        points3d[i, :] = triangulateOnePoint(track, cameraMatrices)

    # TODO: calculate the reprojection errors
    # errors = reprojectionErrors(points3d, cameraMatrices, pointTracks)

    return points3d


# Test case
if __name__ == '__main__':
    # triangulateMultiView()
    pass