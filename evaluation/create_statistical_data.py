import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import transforms3d

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.data_utils import RegistrationData, ModelNet40Data, KittiDataset
from learning3d.ops import se3

# Class to apply transformations.
class PNLKTransform:
	""" rigid motion """
	def __init__(self, mag=1, mag_randomly=False):
		self.mag = mag 									# Angle in radians.
		self.randomly = mag_randomly 					# Random angles < mag.

		# Choose random translation in certain range or in [-0.3, 0.3]
		if not self.randomly: self.max_trans = 0.3
		else: self.max_trans = self.mag

		self.gt = None 				# Source -> Template
		self.igt = None 			# Template -> Source

	def generate_transform(self):
		# return: a twist-vector
		amp = self.mag

		if self.randomly:
			amp = torch.rand(1, 1) * self.mag

		# Create vector for rotation.
		x = torch.randn(1, 3)
		x = x / x.norm(p=2, dim=1, keepdim=True) * amp

		# Create vector for translation.
		t = torch.rand(1, 3) * self.max_trans
		x = torch.cat((x, t), dim=1)

		return x # [1, 6]

	def apply_transform(self, p0, x):
		# p0: 			Template [N, 3]
		# x: 			Twist Vector [1, 6]
		# p1:			Source [N, 3]

		g = se3.exp(x).to(p0)   # [1, 4, 4]
		gt = se3.exp(-x).to(p0) # [1, 4, 4]
		p1 = se3.transform(g, p0[:, :3]) 		# p1 = Tgt * p0

		if p0.shape[1] == 6:  # Need to rotate normals also
			g_n = g.clone()
			g_n[:, :3, 3] = 0.0
			n1 = se3.transform(g_n, p0[:, 3:6])
			p1 = torch.cat([p1, n1], axis=-1)

		self.gt = gt.squeeze(0) 		#  gt: p1 -> p0
		self.igt = g.squeeze(0) 		# igt: p0 -> p1
		return p1

	def transform(self, tensor):
		# tensor: 		Point Cloud [N, 3]
		# x: 			Twist Vector [1, 6]

		x = self.generate_transform()
		return self.apply_transform(tensor, x)

	def __call__(self, tensor):
		# tensor: 		Point Cloud [N, 3]
		return self.transform(tensor)

# Create partial point cloud by sampling knn points near a random point.
def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	# pointcloud1: 			Point Cloud [N, 3] (tensor or ndarray)
	# gt_mask:				Mask [N, 1] (tensor)

	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]

	# obj to find nearest neigbours.
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	
	# Chose random point to find "num_subsampled_points" points near it.
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])

	# Find indices of nearest neighbours for the random point.
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1) 							# Create a mask of 1's or 0's
	return pointcloud1[idx1, :], gt_mask

def add_outliers(pointcloud, gt_mask, size=100):
	# pointcloud: 			Point Cloud (ndarray) [NxC]
	# size: 				Number of outliers to be added (int)
	# output: 				Corrupted Point Cloud (ndarray) [(N+300)xC]
	
	if size == 0: return pointcloud, gt_mask			# If size is 0 then, don't do anything.

	if not torch.is_tensor(pointcloud): pointcloud = torch.tensor(pointcloud)
	N, C = pointcloud.shape
	outliers = 2*torch.rand(size, C)-1 					# Sample points in a cube [-0.5, 0.5]
	pointcloud = torch.cat([pointcloud, outliers], dim=0)
	gt_mask = torch.cat([gt_mask, torch.zeros(size)])

	idx = torch.randperm(pointcloud.shape[0])
	pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]

	return pointcloud.detach().cpu().numpy(), gt_mask

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)

# Visualize data points.
def visualize_result(template, source, est_T):
	template, source, est_T = template, source, est_T
	transformed_template = np.matmul(est_T[0:3, 0:3], template.T).T + est_T[0:3, 3]		# Rotate template as per inverse ground truth est_T.

	template = pc2open3d(template)
	source = pc2open3d(source)
	transformed_template = pc2open3d(transformed_template)
	
	template.paint_uniform_color([1, 0, 0])
	source.paint_uniform_color([0, 1, 0])
	transformed_template.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([template, source, transformed_template])

# Add data in h5py file.
def save_dataset(testdata, group_name, file):
	# testdata:			data to be stored (list of list)
	# group_name:		group in h5py hierarchy
	# file:				h5py File object

	group = file.create_group(group_name)							# Create a group inside h5py File.
	t_list, s_list, igt_list, mask_list = testdata  				# seperate lists.
	group.create_dataset('templates', data=np.array(t_list)) 		# create_dataset with name 'templates' and associated data.
	group.create_dataset('sources', data=np.array(s_list))
	group.create_dataset('igt', data=np.array(igt_list))
	group.create_dataset('masks', data=np.array(mask_list))
	return file

# Create random axes for axangle transformation.
def create_axes(size=1000):
	axes = []
	for _ in range(size):
		axis = np.random.randn(1,3)
		axis = axis/np.sqrt(np.mean(np.square(axis)))
		axes.append(axis[0])
	return np.array(axes)

# Add noise in point cloud.
def jitter_pointcloud(pointcloud, sigma=0.0):
	# pointcloud:		[B, N, 3] or [N, 3] (ndarray)

	pointcloud = torch.tensor(pointcloud).float()
	if sigma > 0:
		pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma)
	return pointcloud.detach().cpu().numpy()

# Create random translations.
def create_translations(size=1000):
	return np.array([np.random.rand(3) for _ in range(size)])

# Define range of angles to be tested.
def angle_range():
	return np.arange(0, 100, 10)

# Define range of noises to be tested.
def noise_range():
	return [0, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2]

# Define range of noises to be tested.
def outlier_range():
	return np.arange(0, 100, 10).tolist() + np.arange(100, 1100, 100).tolist()

# Function to apply ax-angle transformation on template.
def transform_axangle(template, axis, angle, translation):
	# template, source:	Point Cloud [N, 3] (ndarray)
	# axis:				Axis of rotation [3,] (ndarray)
	# angle:			Magnitude of rotation in radians (scalar)
	# translation:		Vector for translation [3, ] (ndarray)
	# igt:				Inverse ground truth transformation [4, 4] (ndarray) (template -> source.)

	igt = np.eye(4)
	igt[0:3, 0:3] = transforms3d.axangles.axangle2mat(axis, angle)
	igt[0:3, 3] = translation
	source = np.matmul(igt[0:3, 0:3], template.T).T + igt[0:3, 3]
	return source, igt

# Function to appply twist-vector transformation on template.
def transform_pnlk(template, transform):
	# template, source:		Point Cloud [N, 3] (ndarray)
	# transform:			object of PNLKTransform class.
	# igt:					Inverse ground truth transformation [4, 4] (ndarray) (template -> source.)

	source = transform(template)
	igt = transform.igt
	return source.detach().cpu().numpy(), igt.detach().cpu().numpy()

# Create a dataset for a particular angle or a noise value.
def create_one_dataset(args, test_loader, axes, angle, translations, noise=0.0, outlier_size=100):
	# args:				Arguments used to create dataset.
	# test_loader:		Contains template point clouds (object of RegistrationData class)
	# axes:				Axes of rotation [B, 3] (ndarray)
	# angle:			Magnitude of angle (scalar)
	# translations:		Translation vectors [B, 3] (ndarray)
	# noise:			Std. Dev. of noise in source point cloud.

	t_list, s_list, igt_list, mask_list = [], [], [], [] 					# templates, sources, inverse gt transformations, masks for partial sources.
	count = 0

	if args.case == 'various_noises': mag_randomly = True 					# If testing for noise levels then random angles.
	elif args.case == 'various_rotations': mag_randomly = False 			# If testing for misalignment levels then fix angle.
	elif args.case == 'various_outliers': mag_randomly = True 				# If testing for outliers then random angles.
	transform = PNLKTransform(mag=angle, mag_randomly=mag_randomly)

	for i, data in enumerate(tqdm(test_loader)):
		if count > args.data_size - 1:
			print("\n")
			break 															# break as the dataset size is reached.
		
		if args.transform_type == 'axis_angle':
			# For Axis Angle Transfomation.
			data_ = [d.detach().cpu().numpy() for d in data]
			template = data_[0]
			source, igt = transform_axangle(template, axes[i], angle, translations[i]) 		# source = T * template

		elif args.transform_type == 'twist_vector':
			# For PNLK Transformation.
			template = data[0]
			source, igt = transform_pnlk(template, transform)								# source = T * template
			template = template.detach().cpu().numpy()

		gt_mask = torch.ones(template.shape[0])												# by default all ones.
		if args.case == 'various_rotations' or args.case == 'various_noises':
			source, gt_mask = farthest_subsample_points(source)			# Create partial source.
		if args.case == 'various_outliers':
			template, gt_mask = add_outliers(template, gt_mask, size=outlier_size)		# Add outliers in template.
		source = jitter_pointcloud(source, sigma=noise)										# Add noise to source.

		# Add data points to lists.
		t_list.append(template)
		s_list.append(source)
		igt_list.append(igt)
		mask_list.append(gt_mask.detach().cpu().numpy())

		# visualize_result(template, source, igt)								# Uncomment to visualize data point.
		count += 1
	return t_list, s_list, igt_list, mask_list

# Create entire dataset.
def create(args, test_loader):
	# args:				Arguments used to create dataset.
	# test_loader:		Contains template point clouds (object of RegistrationData class)

	axes = create_axes(args.data_size) 							# create axes of rotation.
	translations = create_translations(args.data_size)			# create translation vectors.

	import h5py
	file = h5py.File(args.name, 'w') 							# create h5py dataset file.

	# If testing for initial misalignment levels.
	if args.case == 'various_rotations':
		for angle in angle_range():
			print("Angle: {} in degrees".format(angle))
			testdata = create_one_dataset(args, test_loader, axes, angle*(np.pi/180), translations)		# create dataset.
			group_name = 'angle_'+str(angle)															# create group name.
			file = save_dataset(testdata, group_name, file)												# save data under group.

	# If testing for noise levels.
	elif args.case == 'various_noises':
		angle = 20								# max angular misalignment.
		for noise in noise_range():
			print("Noise Level: {}".format(noise))
			testdata = create_one_dataset(args, test_loader, axes, angle*(np.pi/180), translations, noise=noise)	# create dataset.
			group_name = 'noise_'+str(noise)																		# create group name.
			file = save_dataset(testdata, group_name, file)															# save data under group.

	# If testing for various outliers in template.
	elif args.case == 'various_outliers':
		angle = 45
		for outliers in outlier_range():
			print("Size of outliers: {}".format(outliers))
			testdata = create_one_dataset(args, test_loader, axes, angle*(np.pi/180), translations, outlier_size=outliers)	# create dataset.
			group_name = 'outliers_'+str(outliers)
			file = save_dataset(testdata, group_name, file)

	file.close()

def create_dataset(args, test_loader):
	create(args, test_loader)

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')

	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for on training
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--version', default='v0', type=str,
						help='Specify version of SampleNet')

	parser.add_argument('--name', default='testdata_various_angles.h5', type=str,
						help='Specify name of file for testdata')
	parser.add_argument('--data_size', default=1000, type=int,
						help='Specify size of dataset')

	# Settings for data.
	parser.add_argument('--case', default='various_rotations', type=str,
						help='Choose which dataset to create', choices=['various_rotations', 'various_noises', 'various_outliers'])
	parser.add_argument('--transform_type', default='twist_vector', type=str,
						help='Choose which type of transform to be used.', choices=['twist_vector', 'axis_angle'])

	# Useful for generalization experiments.
	parser.add_argument('--unseen', default=False, type=bool,
						help='False: Use all 40 categories and True: Split data into first and last 20 categories.')
	parser.add_argument('--train_data', default=False, type=bool,
						help='True for first 20 categories and False for last 20 categories')

	args = parser.parse_args()
	return args

def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	
	trainset = RegistrationData(ModelNet40Data(train=True, num_points=args.num_points, unseen=args.unseen, use_normals=False))
	testset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, unseen=args.unseen, use_normals=False))

	if args.unseen:
		if args.train_data: test_loader = trainset
		else: test_loader = testset
	else: 
		test_loader = testset

	# Check for size of dataset.
	assert args.data_size <= len(test_loader), "Given invalid data_size (> size of test_loader)"

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	create_dataset(args, test_loader)

if __name__ == '__main__':
	main()