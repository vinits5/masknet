import open3d as o3d
import argparse
import os
import sys
import copy
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import MaskNet
from learning3d.data_utils import RegistrationData, ModelNet40Data, UserData, AnyData
from registration import Registration

def farthest_point_sample(xyz, npoint):
	"""
	Input:
		xyz: pointcloud data, [B, N, C]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [B, npoint]
	"""
	#import ipdb; ipdb.set_trace()
	if not torch.is_tensor(xyz): xyz = torch.tensor(xyz).float().view(1, -1, 3)
	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	batch_indices = torch.arange(B, dtype=torch.long).to(device)
	for i in range(npoint):
		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]
	return centroids.cpu().numpy()[0]

def normalize_pc(point_cloud):
	centroid = np.mean(point_cloud, axis=0)
	point_cloud -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud)**2,axis=-1)))
	point_cloud /= furthest_distance
	return point_cloud

def read_mesh(path, sample_pc=True, num_points=10000):
	pc = o3d.io.read_point_cloud(path)
	points = normalize_pc(np.array(pc.points))
	
	if sample_pc:
		# points_idx = farthest_point_sample(points, 10000)
		points_idx = np.arange(points.shape[0])
		np.random.shuffle(points_idx)
		points = points[points_idx[:10000], :]
	return points

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)

# This function is taken from Deep Global Registration paper's github repository.
def create_pcd(xyz, color):
	# n x 3
	n = xyz.shape[0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	return pcd

# This function is taken from Deep Global Registration paper's github repository.
def draw_geometries_flip(pcds):
	pcds_transform = []
	flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
	for pcd in pcds:
		pcd_temp = copy.deepcopy(pcd)
		pcd_temp.transform(flip_transform)
		pcds_transform.append(pcd_temp)
	o3d.visualization.draw_geometries(pcds_transform)

def display_results(template, source, est_T, mask_idx):
	non_mask_idx = np.array([i for i in range(mask_idx.shape[0]) if i not in mask_idx])
	unmasked_template = template[non_mask_idx]
	masked_template = template[mask_idx]

	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]
	
	template = create_pcd(template, np.array([1, 0.706, 0]))
	source = create_pcd(source, np.array([0, 0.929, 0.651]))
	transformed_source = create_pcd(transformed_source, np.array([0, 0.651, 0.921]))
	masked_template = create_pcd(masked_template, np.array([0,0,1]))
	unmasked_template = create_pcd(unmasked_template, np.array([1,0,0]))

	draw_geometries_flip([template, source])
	draw_geometries_flip([template, transformed_source])
	draw_geometries_flip([masked_template, unmasked_template])

def store_results(args, template, source, est_T, igt, gt_mask, predicted_mask, mask_idx, est_T_series):
	est_T_series = est_T_series.detach().cpu().numpy().reshape(-1, 4, 4)
	est_T_series = est_T_series.reshape(-1, 4)
	mesh = read_mesh(path=args.dataset_path, sample_pc=False)

	np.savez('3dmatch_results', template=template.detach().cpu().numpy()[0], 
			  source = source.detach().cpu().numpy()[0], est_T = est_T.detach().cpu().numpy()[0], 
			  igt = igt.detach().cpu().numpy()[0], gt_mask=gt_mask.detach().cpu().numpy()[0], 
			  mask_idx=mask_idx.detach().cpu().numpy()[0], est_T_series=est_T_series,
			  predicted_mask=predicted_mask.detach().cpu().numpy()[0], mesh=mesh)

def evaluate_metrics(TP, FP, FN, TN, gt_mask):
	# TP, FP, FN, TN: 		True +ve, False +ve, False -ve, True -ve
	# gt_mask:				Ground Truth mask [Nt, 1]
	
	accuracy = (TP + TN)/gt_mask.shape[1]
	misclassification_rate = (FN + FP)/gt_mask.shape[1]
	# Precision: (What portion of positive identifications are actually correct?)
	precision = TP / (TP + FP)
	# Recall: (What portion of actual positives are identified correctly?)
	recall = TP / (TP + FN)

	fscore = (2*precision*recall) / (precision + recall)
	return accuracy, precision, recall, fscore

# Function used to evaluate the predicted mask with ground truth mask.
def evaluate_mask(gt_mask, predicted_mask, predicted_mask_idx):
	# gt_mask:					Ground Truth Mask [Nt, 1]
	# predicted_mask:			Mask predicted by network [Nt, 1]
	# predicted_mask_idx:		Point indices chosen by network [Ns, 1]

	if torch.is_tensor(gt_mask): gt_mask = gt_mask.detach().cpu().numpy()
	if torch.is_tensor(gt_mask): predicted_mask = predicted_mask.detach().cpu().numpy()
	if torch.is_tensor(predicted_mask_idx): predicted_mask_idx = predicted_mask_idx.detach().cpu().numpy()
	gt_mask, predicted_mask, predicted_mask_idx = gt_mask.reshape(1,-1), predicted_mask.reshape(1,-1), predicted_mask_idx.reshape(1,-1)
	
	gt_idx = np.where(gt_mask == 1)[1].reshape(1,-1) 				# Find indices of points which are actually in source.

	# TP + FP = number of source points.
	TP = np.intersect1d(predicted_mask_idx[0], gt_idx[0]).shape[0]			# is inliner and predicted as inlier (True Positive) 		(Find common indices in predicted_mask_idx, gt_idx)
	FP = len([x for x in predicted_mask_idx[0] if x not in gt_idx])			# isn't inlier but predicted as inlier (False Positive)
	FN = FP															# is inlier but predicted as outlier (False Negative) (due to binary classification)
	TN = gt_mask.shape[1] - gt_idx.shape[1] - FN 					# is outlier and predicted as outlier (True Negative)
	return evaluate_metrics(TP, FP, FN, TN, gt_mask)

def test_one_epoch(args, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	AP_List = []
	GT_Size_List = []
	precision_list = []
	registration_model = Registration(args.reg_algorithm)

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)						# [source] = [igt]*[template]
		gt_mask = gt_mask.to(args.device)

		masked_template, predicted_mask = model(template, source)
		result = registration_model.register(masked_template, source)
		est_T = result['est_T']
		est_T_series = result['est_T_series']			#[11, 1, 4, 4]
		
		# Evaluate mask based on classification metrics.
		accuracy, precision, recall, fscore = evaluate_mask(gt_mask, predicted_mask, predicted_mask_idx = model.mask_idx)
		precision_list.append(precision)

		if args.store_results: store_results(args, template, source, est_T, igt, gt_mask, predicted_mask, model.mask_idx, est_T_series)
		
		# Different ways to visualize results.
		display_results(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], est_T.detach().cpu().numpy()[0], model.mask_idx.detach().cpu().numpy()[0])

	print("Mean Precision: ", np.mean(precision_list))

def test(args, model, test_loader):
	test_one_epoch(args, model, test_loader)

def options():
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (3DMatch Testing)')
	
	# settings for input data
	parser.add_argument('--dataset_path', default='../learning3d/data/sun3d-home_at-home_at_scan1_2013_jan_1/cloud_bin_29.ply', type=str,
						help='Provide the path to .ply file in 3DMatch dataset.')
	parser.add_argument('--num_points', default=10000, type=int, help='Number of points in sampled point cloud')
	parser.add_argument('--partial_source', default=True, type=bool,
						help='create partial source point cloud in dataset.')
	parser.add_argument('--noise', default=False, type=bool,
						help='Add noise in source point clouds.')
	parser.add_argument('--outliers', default=False, type=bool,
						help='Add outliers to template point cloud.')

	# settings for on testing
	parser.add_argument('-j', '--workers', default=1, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--test_batch_size', default=1, type=int,
						metavar='N', help='test-mini-batch size (default: 1)')
	parser.add_argument('--reg_algorithm', default='pointnetlk', type=str,
						help='Algorithm used for registration.', choices=['pointnetlk', 'icp', 'dcp', 'prnet', 'pcrnet', 'rpmnet'])
	parser.add_argument('--pretrained', default='../pretrained/model_masknet_3DMatch.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--store_results', default=True, type=bool,
						help='Store results of 3DMatch Test')

	args = parser.parse_args()
	return args

def main():
	args = options()
	torch.backends.cudnn.deterministic = True

	points = read_mesh(path=args.dataset_path, sample_pc=True, num_points=args.num_points)
	testset = AnyData(pc=points, mask=True, repeat=1)
	test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Load Pretrained MaskNet.
	model = MaskNet()
	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model = model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()