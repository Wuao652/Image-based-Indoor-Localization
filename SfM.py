import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utils.cameraParams import generateIntrinsics


img1 = cv2.imread('./data/image1.jpg')
img2 = cv2.imread('./data/image2.jpg')
img3 = cv2.imread('./data/image3.jpg')
img4 = cv2.imread('./data/image4.jpg')
img5 = cv2.imread('./data/image5.jpg')

def plot_imgs():
    fig, ax = plt.subplots(3, 2)
    for a in ax.reshape((-1)):
      a.set_axis_off()
    ax[0][0].imshow(img1)
    ax[0][1].imshow(img2)
    ax[1][0].imshow(img3)
    ax[1][1].imshow(img4)
    ax[2][0].imshow(img5)
    plt.show()
# plot_imgs()


if __name__ == '__main__':
    print("hello world from SfM!")
    intrinsics = generateIntrinsics()
    # matlab_img = scipy.io.loadmat('./data/images.mat')
    # matlab_img = matlab_img['images']

    # Convert the images to grayscale.
    images = []
    for i in [img1, img2, img3, img4, img5]:
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        images.append(gray)
    sift = cv2.xfeatures2d.SIFT_create()
    # Find keypoints and descriptors directly
    kp, des = sift.detectAndCompute(images[0], None)
    print(len(kp))








