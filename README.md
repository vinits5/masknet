# MaskNet: A Fully-Convolutional Network to Estimate Inlier Points

Source Code Author: Vinit Sarode

This work is accepted at the 8th International Conference on 3D Vision.

### Binary mask prediction using MaskNet:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/approach.gif" height="300">
</p>

### Applications:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/applications.png" height="200">
</p>

### Network Architecture:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/network.png" height="300">
</p>

### Requirements:
1. pytorch==1.3.0+cu92
2. transforms3d==0.3.1
3. h5py==2.9.0
4. open3d==0.7.0.0
5. ninja==1.9.0.post1
6. tensorboardX=1.8

### Learning3D: A Modern Library for Deep Learning on 3D Point Clouds Data
Learning3D is our open-source library that supports the development of deep learning algorithms that deal with 3D data. The Learning3D exposes a set of state of art deep neural networks in python. A modular code has been provided for further development. We welcome contributions from the open-source community.

[CODE](https://github.com/vinits5/learning3d) | [DOCUMENTATION](https://medium.com/@vinitsarode5/learning3d-a-modern-library-for-deep-learning-on-3d-point-clouds-data-48adc1fd3e0?sk=0beb59651e5ce980243bcdfbf0859b7a) | [DEMO](https://github.com/vinits5/learning3d/blob/master/examples/test_pointnet.py)

### Results of 3DMatch Dataset:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/3.gif" height="200">
	<img src="https://github.com/vinits5/masknet/blob/main/images/4.gif" height="200">
</p>

<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/1.gif" height="200">
	<img src="https://github.com/vinits5/masknet/blob/main/images/2.gif" height="200">
</p>

## Use of Code

### Dataset:
> ./learning3d/data_utils/download_data.sh

### Train MaskNet:
> conda create -n masknet python=3.7
> pip install -r requirements.txt
> python train.py

### Test MaskNet:
> python test.py --pretrained checkpoints/exp_masknet/models/best_model.t7 --reg_algorithm 'pointnetlk'\

We provide a number of registration algorithms with MaskNet as listed below:
1. PointNetLK
2. Deep Closest Point (DCP)
3. Iterative Closest Point (ICP)
4. PRNet
5. PCRNet
6. RPMNet

### Statistical Results:
> cd evaluation && chmod +x evaluate.sh && ./evaluate.sh

### Tests with 3D-Match:
> python download_3dmatch.py\
> python test_3DMatch.py\
> python plot_figures.py\
> python make_video.py

### License
MIT License


We would like to thank the authors of [PRNet](https://papers.nips.cc/paper/9085-prnet-self-supervised-learning-for-partial-to-partial-registration.pdf), [PointNetLK](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf) for sharing their codes.