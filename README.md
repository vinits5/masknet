# MaskNet: A Fully-Convolutional Network to Estimate Inlier Points
[**Paper**](https://arxiv.org/abs/2010.09185) | [**Website**]() | [**Video**](https://youtu.be/sOubGRB17D4) | [**Biorobotics Lab's Project Webpage**](http://biorobotics.ri.cmu.edu/research/ml_registration.php)

Source Code Author: Vinit Sarode

*[Note: This work is accepted at the 8th International Conference on 3D Vision, 2020.]*

### Binary mask prediction using MaskNet:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/approach.gif" height="300">
</p>

In this work, MaskNet estimates inliers (shown on the right in blue) for a given pair of point clouds (shown on the left). MaskNet finds a Boolean vector mask that only retains inlier points from point cloud in red which most closely approximate the shape of the point cloud in green.

We call our method MaskNet as the network learns to 'mask-out' outliers from template (shown in red) point cloud. We demonstrate the efficiency of MaskNet as a pre-processing step in various applications (as shown below). MaskNet shows remarkable generalization within and across datasets without the need for additional fine-tuning.

### Applications:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/applications.png" height="200">
</p>

### Results of 3DMatch Dataset:
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/3.gif" height="200">
	<img src="https://github.com/vinits5/masknet/blob/main/images/4.gif" height="200">
</p>

<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/1.gif" height="200">
	<img src="https://github.com/vinits5/masknet/blob/main/images/2.gif" height="200">
</p>

### Citation:
```
@misc{sarode2020masknet,
      title={MaskNet: A Fully-Convolutional Network to Estimate Inlier Points}, 
      author={Vinit Sarode and Animesh Dhagat and Rangaprasad Arun Srivatsan and Nicolas Zevallos and Simon Lucey and Howie Choset},
      year={2020},
      eprint={2010.09185},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Learning3D: A Modern Library for Deep Learning on 3D Point Clouds Data
Learning3D is our open-source library that supports the development of deep learning algorithms that deal with 3D data. The Learning3D exposes a set of state of art deep neural networks in python. A modular code has been provided for further development. We welcome contributions from the open-source community.\
*[Note: We have used learning3d library while implementing MaskNet. Feel free to refer to following references.]*

[**Code**](https://github.com/vinits5/learning3d) | [**Documentation**](https://medium.com/@vinitsarode5/learning3d-a-modern-library-for-deep-learning-on-3d-point-clouds-data-48adc1fd3e0?sk=0beb59651e5ce980243bcdfbf0859b7a) | [**Demo**](https://github.com/vinits5/learning3d/blob/master/examples/test_pointnet.py)

## Usage

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

### Dataset:
> ./learning3d/data_utils/download_data.sh

### Train MaskNet:
> conda create -n masknet python=3.7\
> pip install -r requirements.txt\
> python train.py

### Test MaskNet:
> python test.py --pretrained checkpoints/exp_masknet/models/best_model.t7 --reg_algorithm 'pointnetlk'

We provide a number of registration algorithms with MaskNet as listed below:
1. PointNetLK
2. Deep Closest Point (DCP)
3. Iterative Closest Point (ICP)
4. PRNet
5. PCRNet
6. RPMNet

### Test MaskNet with Your Own Data:
In the test.py file, change the template and source variables with your data on line number 156 and 157. Ground truth values for mask and transformation between template and source can be provided by changing the variables on line no. 158 and 159 resp. 
> python test.py --user_data True --reg_algorithm 'pointnetlk'

### Statistical Results:
> cd evaluation && chmod +x evaluate.sh && ./evaluate.sh

### Tests with 3D-Match:
> python download_3dmatch.py\
> python test_3DMatch.py\
> python plot_figures.py\
> python make_video.py

### License
This project is release under the MIT License.


We would like to thank the authors of [PointNet](http://stanford.edu/~rqi/pointnet/), [PRNet](https://papers.nips.cc/paper/9085-prnet-self-supervised-learning-for-partial-to-partial-registration.pdf), [RPM-Net](https://arxiv.org/abs/2003.13479) and [PointNetLK](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf) for sharing their codes.


## Additional Results:

**3D-Match Dataset Results:**
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/3dmatch_results.gif" height="300">
</p>

**Iterative registration with MaskNet:**
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/iterations_registration.png" height="300">
</p>

**MaskNet's Sensitivity to Noise:**
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/noise_sensitivity.png" height="300">
</p>

**Point Cloud Denoising:**
<p align="center">
	<img src="https://github.com/vinits5/masknet/blob/main/images/denoising.png" height="300">
</p>
