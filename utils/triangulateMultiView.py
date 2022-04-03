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
    V = V.T
    X = V[:, -1]
    X = X/X[-1]

    return X[0:3]


def reprojectPoint(p3dh, viewIds, cameraMatrices):
    numPoints = len(viewIds)
    points2d = np.zeros((numPoints, 2))
    for i in range(numPoints):
        p2dh = p3dh @ cameraMatrices[viewIds[i]]
        points2d[i, :] = p2dh[0:2]/p2dh[2]

    return points2d


def reprojectionErrors(points3d, cameraMatrices, tracks):
    numPoints = points3d.shape[0]
    points3dh = np.hstack([points3d, np.ones((numPoints, 1))])
    meanErrorsPerTrack = np.zeros((numPoints, 1))
    errors = []
    for i in range(numPoints):
        p3d = points3dh[i, :]

        track = list(tracks.values())[i]
        viewIds = np.array([e[0] for e in track])
        points  = np.array([e[1] for e in track])

        reprojPoints2D = reprojectPoint(p3d, viewIds, cameraMatrices)
        e = np.sqrt(((points - reprojPoints2D)**2).sum(1))
        # print(e)
        meanErrorsPerTrack[i] = e.mean()
        errors.append(e)

    return errors, meanErrorsPerTrack



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

    Output is a numpy array with all the 3D points in order.
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
    _, errors = reprojectionErrors(points3d, cameraMatrices, pointTracks)

    return points3d, errors


# Test case
if __name__ == '__main__':
    # triangulateMultiView()
    pass