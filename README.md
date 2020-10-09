# MaskNet: A Fully-Convolutional Network to Estimate Inlier Points

Source Code Author: Vinit Sarode

### Binary mask prediction using MaskNet:
<p align="center">
	<img src="https://github.com/vinits5/masknet/tree/main/images/approach.gif" height="300">
</p>

### Network Architecture:
<p align="center">
	<img src="https://github.com/vinits5/masknet/tree/main/images/network.png" height="300">
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
> conda create -n masknet python=3.7
> pip install -r requirements.txt
> python train.py

### Test MaskNet:
> python test.py --pretrained checkpoints/exp_masknet/models/best_model.t7

### Statistical Results:
> cd evaluation && chmod +x evaluate.sh && ./evaluate.sh

### Tests with 3D-Match:
> python download_3dmatch.py\
> python test_3DMatch.py\
> python plot_figures.py\
> python make_video.py

### Results of 3DMatch Dataset:
<p align="center">
	<img src="https://github.com/vinits5/masknet/tree/main/images/3.gif" height="300">
	<img src="https://github.com/vinits5/masknet/tree/main/images/4.gif" height="300">
</p>

<p align="center">
	<img src="https://github.com/vinits5/masknet/tree/main/images/1.gif" height="300">
	<img src="https://github.com/vinits5/masknet/tree/main/images/2.gif" height="300">
</p>