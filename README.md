# Image-based Indoor Localization
This project is a faithful Tensorflow implementation of the paper. The code is based on the authors' original implementation and has been tested to match it numerically.

Here is the link to the original [paper](https://arxiv.org/abs/2201.01408)

## Table of Contents
- [Image-based Indoor Localization](#image-based-indoor-localization)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Download Dataset](#download-dataset)
  - [How to Run?](#how-to-run)
  - [Some of the Results](#some-of-the-results)
    - [Keypoint matching](#keypoint-matching)
    - [Final estimation](#final-estimation)
  - [Conclusion](#conclusion)
  - [Authors](#authors)


## Background
This paper proposes a new image-based localization framework that explicitly localizes the camera/robot by fusing Convolutional Neural Network (CNN) and sequential images' geometric constraints. The camera is localized using a single or few observed images and training images with 6-degree-of-freedom pose labels. A Siamese network structure is adopted to train an image descriptor network, and the visually similar candidate image in the training set is retrieved to localize the testing image geometrically. Meanwhile, a probabilistic motion model predicts the pose based on a constant velocity assumption. The two estimated poses are finally fused using their uncertainties to yield an accurate pose prediction. This method leverages the geometric uncertainty and is applicable in indoor scenarios predominated by diffuse illumination. Experiments on simulation and real data sets demonstrate the efficiency of our proposed method. The results further show that combining the CNN-based framework with geometric constraint achieves better accuracy when compared with CNN-only methods, especially when the training data size is small.

## Overview
The original paper fuses the CNN model and the probabilistic motion model to predict the camera pose, which is shown in the following figure. For us, we implement the CNN model in python, which is shown in the dashed line.
![](./images/relocate.png)


## Installation
The code is totally in python, the set up for this repo is very easy. 
```
git clone https://github.com/Wuao652/Image-based-Indoor-Localization.git
cd Image-based-Indoor-Localization
```
For running this package, you need the following dependences (just use pip or conda to install them),
- `numpy, scipy` for dealing math and array calculation.
- `siamese_zone/tf2-gpu.yml` can help create a conda environment for CNN model training.
- `h5py` for reading pretrained CNN model.
- `matplotlib` for ploting.
- `opencv-contrib-python` for image processing.


## Download Dataset
Currently, this package can only handle the TUM dataset, for example `1_desk2` subset, you can download the data [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_desk2).
Once you download the dataset, you have to put it under the folder `data/TUM/1_desk2`. If you don't have such a folder, just make a new one by yourself! You may also need the pre-trained CNN model, please contact the contributor for a pre-trained model or you can train the model by your self too!


## How to Run?
After installing all the depedences and download the dataset, you can just run
```
python SfM.py
```

## Some of the Results
### Keypoint matching
This part shows an example of what the keypoint matching looks like,
![](./images/kfs_1.png)

The following figure shows the selected keypoint,
![](./images/kfs_2.png)

The following figure shows the final matching result,
![](./images/kfs_3.png)

### Final estimation
This plot shows the final estimation result for the `desk2` subset of `TUM` dataset,
![](./images/3d_estimation.png)


## Conclusion
- While single GL is good enough to produce high accuracy pose estimation, we still need motion model and fusing algorithm to deal with the failure case and images mismatch result from CNN.
- The method requires small dataset and outperforms image-to-pose methods by overcoming issues caused by differences in viewing angles between the testing and training datasets.
- Limitations:
  - The geometric locator is unreliable if there is different illumination between the training and test image
  - This approach requires knowledge of the camera intrinsic parameters


## Authors
Wuao Liu, Man Yuan, Wan-Yi Yu, Sairub Naaz, Tamaira Linares
