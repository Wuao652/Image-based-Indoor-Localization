# SfM-demo
This project is a faithful Tensorflow implementation of the paper. The code is based on the authors' original implementation and has been tested to match it numerically.

Here is the link to the original [paper](https://arxiv.org/abs/2201.01408)

## Installation
```
git clone https://github.com/Wuao652/SfM-demo.git
cd SfM-demo
```
For running this package, you need the following dependences,
```
numpy
scipy
h5py
matplotlib
opencv-contrib-python
```

## Download Dataset
Currently this package can only handle the TUM dataset `1_desk2` subset, you can download the data [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_desk2).
Once you donwload the dataset, you have to put it under the folder `data/TUM/1_desk2`. If you don't have such a folder, just make a new one by your self!

## How to Run?
After installing all the depedences and download the dataset, you can just run
```
python SfM.py
```

## Some of the Results
### Keypoint matching
### Triangulation
### Final estimation
![](./images/3d_estimation.png)