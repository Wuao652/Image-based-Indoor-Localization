import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utils.cameraParams import generateIntrinsics
img1 = plt.imread('./data/image1.jpg')
img2 = plt.imread('./data/image2.jpg')
img3 = plt.imread('./data/image3.jpg')
img4 = plt.imread('./data/image4.jpg')
img5 = plt.imread('./data/image5.jpg')

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





