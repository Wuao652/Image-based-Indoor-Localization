def generateIntrinsics():
    # intrinsics = {}
    # intrinsics['FocalLength'] = [1037.575214664696, 1043.315752317925]
    # intrinsics['PrincipalPoint'] = [642.2315830312182, 387.8357750962377]
    # intrinsics['ImageSize'] = [720, 1280]
    # intrinsics['RadialDistortion'] = [0.146911684283474, -0.214389634520344]
    # intrinsics['TangentialDistortion'] = [0., 0.]
    # intrinsics['Skew'] = 0.
    # intrinsics['IntrinsicMatrix'] = [[1037.575214664696,    0.               ,      0.],
    #                                  [0.               ,    1043.315752317925,      0.],
    #                                  [642.2315830312182,	387.8357750962377,  	1.]]

    intrinsics = {}
    intrinsics['FocalLength'] = [517.300000000000, 516.500000000000]
    intrinsics['PrincipalPoint'] = [318.600000000000, 255.300000000000]
    intrinsics['ImageSize'] = []
    intrinsics['RadialDistortion'] = [0., 0., 0.]
    intrinsics['TangentialDistortion'] = [0., 0.]
    intrinsics['Skew'] = 0.
    intrinsics['IntrinsicMatrix'] = [[517.300000000000,    0.              ,      0.],
                                     [0.              ,    516.500000000000,      0.],
                                     [318.600000000000,	   255.300000000000,  	  1.]]

    return intrinsics


if __name__ == '__main__':
    intrinsics = generateIntrinsics()