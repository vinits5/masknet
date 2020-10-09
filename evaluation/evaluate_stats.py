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
from tensorboardX import SummaryWriter
from tqdm import tqdm
import transforms3d.euler as t3d
import transforms3d
import time

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import MaskNet, PointNet, iPCRNet, RPMNet, PointNetLK, DGCNN, DCP, PRNet
from learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss
from learning3d.losses import ChamferDistanceLoss
from TestDataLoader import TestDataLoader

def pc2points(data):
	if len(data.shape) == 3:
		return data[:, :, :3]
	elif len(data.shape) == 2:
		return data[:, :3]

# To avoid samplenet.
class Identity(torch.nn.Module):
	def forward(self, *input):
		return input 					# return inputs as it is.

# ICP registration module.
class ICP:
	def __init__(self, threshold=0.1, max_iteration=10):
		# threshold: 			Threshold for correspondences. (scalar)
		# max_iterations:		Number of allowed iterations. (scalar)
		self.threshold = threshold
		self.criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

	# Preprocess template, source point clouds.
	def preprocess(self, template, source):
		if self.is_tensor: template, source = template.detach().cpu().numpy(), source.detach().cpu().numpy()	# Convert to ndarray if tensors.

		if len(template.shape) > 2: 						# Reduce dimension to [N, 3]
			template, source = template[0], source[0]

		# Find mean of template & source.
		self.template_mean = np.mean(template, axis=0, keepdims=True)
		self.source_mean = np.mean(source, axis=0, keepdims=True)
		
		# Convert to open3d point clouds.
		template_ = o3d.geometry.PointCloud()
		source_ = o3d.geometry.PointCloud()

		# Subtract respective mean from each point cloud.
		template_.points = o3d.utility.Vector3dVector(template - self.template_mean)
		source_.points = o3d.utility.Vector3dVector(source - self.source_mean)
		return template_, source_

	# Postprocess on transformation matrix.
	def postprocess(self, res):
		# Way to deal with mean substraction
		# Pt = R*Ps + t 								original data (1)
		# Pt - Ptm = R'*[Ps - Psm] + t' 				mean subtracted from template and source.
		# Pt = R'*Ps + t' - R'*Psm + Ptm 				rearrange the equation (2)
		# From eq. 1 and eq. 2,
		# R = R' 	&	t = t' - R'*Psm + Ptm			(3)

		est_R = np.array(res.transformation[0:3, 0:3]) 						# ICP's rotation matrix (source -> template)
		t_ = np.array(res.transformation[0:3, 3]).reshape(1, -1)			# ICP's translation vector (source -> template)
		est_T = np.array(res.transformation)								# ICP's transformation matrix (source -> template)
		est_t = np.matmul(est_R, -self.source_mean.T).T + t_ + self.template_mean[0] 	# update predicted translation according to eq. 3
		est_T[0:3, 3] = est_t
		return est_R, est_t, est_T

	# Convert result to pytorch tensors.
	@staticmethod
	def convert2tensor(result):
		if torch.cuda.is_available(): device = 'cuda'
		else: device = 'cpu'
		result['est_R']=torch.tensor(result['est_R']).to(device).float().view(-1, 3, 3) 		# Rotation matrix [B, 3, 3] (source -> template)
		result['est_t']=torch.tensor(result['est_t']).to(device).float().view(-1, 1, 3)			# Translation vector [B, 1, 3] (source -> template)
		result['est_T']=torch.tensor(result['est_T']).to(device).float().view(-1, 4, 4)			# Transformation matrix [B, 4, 4] (source -> template)
		return result

	# icp registration.
	def __call__(self, template, source):
		self.is_tensor = torch.is_tensor(template)

		template, source = self.preprocess(template, source)
		res = o3d.registration.registration_icp(source, template, self.threshold, criteria=self.criteria)	# icp registration in open3d.
		est_R, est_t, est_T = self.postprocess(res)
		
		result = {'est_R': est_R,
				  'est_t': est_t,
				  'est_T': est_T}
		if self.is_tensor: result = self.convert2tensor(result)
		return result

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
	# template, source:			Point Clouds [N, 3] (ndarray)
	# est_T: 					Predicted Transformation [4, 4] (ndarray) (source -> template)

	template, source, est_T = template[0], source[0], est_T[0]
	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3] 		# Rotate template as per inverse ground truth est_T.
	
	# Allocate points to each open3d point cloud.
	template = pc2open3d(template)
	source = pc2open3d(source)
	transformed_source = pc2open3d(transformed_source)
	
	# Apply color to each point cloud.
	template.paint_uniform_color([1, 0, 0])
	source.paint_uniform_color([0, 1, 0])
	transformed_source.paint_uniform_color([0, 0, 1])

	# Display point clouds.
	o3d.visualization.draw_geometries([template, source, transformed_source])

# Find error metrics.
def find_errors(gt_R, pred_R, gt_t, pred_t):
	# gt_R:				Rotation matrix [3, 3] (source = gt_R * template)
	# pred_R: 			Registration algorithm's rotation matrix [3, 3] (template = pred_R * source)
	# gt_t:				translation vector [1, 3] (source = template + gt_t)
	# pred_t: 			Registration algorithm's translation matrix [1, 3] (template = source + pred_t)

    # Euler distance between ground truth translation and predicted translation.
    gt_t = -np.matmul(gt_R.T, gt_t) 									# gt translation vector (source -> template)
    translation_error = np.sqrt(np.sum(np.square(gt_t - pred_t)))

    # Convert matrix remains to axis angle representation and report the angle as rotation error.
    error_mat = np.dot(gt_R, pred_R)							# matrix remains [3, 3]
    _, angle = transforms3d.axangles.mat2axangle(error_mat)
    return translation_error, abs(angle*(180/np.pi))

# Evaluate metrics.
def evaluate_results(template, source, est_T, igt):
	# template, source: 		Point Cloud [B, N, 3] (torch tensor)
	# est_T:					Predicted transformation [B, 4, 4] (torch tensor) (template = est_T * source)
	# igt: 						Ground truth transformation [B, 4, 4] (torch tensor) (source = igt * template)

	transformed_source = torch.bmm(est_T[:, 0:3, 0:3], source.permute(0, 2, 1)).permute(0,2,1) + est_T[:, 0:3, 3]
	try:
		cd_loss = ChamferDistanceLoss()(template, transformed_source).item()					# Find chamfer distance between template and registered source.
	except:
		cd_loss = 0.0

	# Find error metrices.
	template, source, est_T, igt = template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], est_T.detach().cpu().numpy()[0], igt.detach().cpu().numpy()[0]
	translation_error, rotation_error = find_errors(igt[:3, :3], est_T[:3, :3], igt[:3, 3], est_T[:3, 3])
	return translation_error, rotation_error, cd_loss

# Register template and source pairs.
def register(template, source, model, reg_algorithm, args):
	# template, source: 		Point Cloud [B, N, 3] (torch tensor)
	# model:					Obj of mask network.
	# reg_algorithm:			Obj of registration algorithm.

	# No need to use normals. Only use normals for RPM-Net.
	if not args.reg_algorithm == 'rpmnet':
		template, source = pc2points(template), pc2points(source)

	if args.masknet:
		masked_template, pred_mask = model(template, source)
		result = reg_algorithm(masked_template, source)
	else:
		template, source = model(template, source) 					# Identity class is used.
		result = reg_algorithm(template, source)
	return result

# Test a dataset using given registration algorithm.
def test_one_epoch(args, model, reg_algorithm, test_loader):
	# args: 			Parameters required for testing.
	# model:			Either obj of mask network or Identity class.
	# reg_algorithm:	Obj of registration algorithm.
	# test_loader:		Obj of test data loading class.

	model.eval()
	device = args.device
	test_loss = 0.0
	pred  = 0.0
	count = 0
	t_errors, r_errors, cd_losses = [], [], []			# lists to store metrics.
	timings = []

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask = data 

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)
		gt_mask = gt_mask.to(device)

		start = time.time()
		result = register(template, source, model, reg_algorithm, args)
		elapsed_time = time.time() - start

		est_T = result['est_T']

		# Function to view result of a data point.
		# visualize_result(template.detach().cpu().numpy(), source.detach().cpu().numpy(), est_T.detach().cpu().numpy())

		t_err, r_err, cd_loss = evaluate_results(pc2points(template), pc2points(source), est_T, igt)
		
		# Add metrics to list.
		t_errors.append(t_err)
		r_errors.append(r_err)
		cd_losses.append(cd_loss)
		timings.append(elapsed_time)

		count += 1

	return t_errors, r_errors, cd_loss, timings

# Store mean and std. dev. in a text file.
def save_stats(t_errors, r_errors, cd_loss, timings, root_dir, group_name=''):
	# t_errors, r_erros, cd_loss, timing:		translation err, rotation err, chamfer distances, elapsed time (list)
	# root_dir:									Path of directory to store results.
	# group_name:								Name of the group in h5py file.

	# Compute mean and std of all metrices.
	r_mean, t_mean, cd_mean, time_mean = np.mean(r_errors), np.mean(t_errors), np.mean(cd_loss), np.mean(timings)
	r_std, t_std, cd_std, time_std = np.std(r_errors), np.std(t_errors), np.std(cd_loss), np.std(timings)
	
	file = open(os.path.join(root_dir, 'results.txt'), 'a')			# Append to a text file.

	# Store mean of results.
	text = 'Mean Rotation Error: {}\nMean Translation Error: {}\nMean CD Loss: {}\nMean Time: {} in sec.'.format(r_mean, t_mean, cd_mean, time_mean)
	if group_name: text = '\n\n' + group_name + '\n' + text 			# for multiple results with various angles.
	file.write(text)
	
	# Store std of results.
	text = '\n\nStd. Dev. Rotation Error: {}\nStd. Dev. Translation Error: {}\nStd. Dev. CD Loss: {}\nStd. Dev. Time: {} in sec.'.format(r_std, t_std, cd_std, time_std)
	file.write(text)
	file.close()

# Store all metrics as lists in h5py file.
def save_results(t_errors, r_errors, cd_loss, timings, name, group_name=''):
	# t_errors, r_erros, cd_loss, timing:		translation err, rotation err, chamfer distances, elapsed time (list)
	# name:										Name of the directory to store results.
	# group_name:								Name of the group in h5py file.

	root_dir = os.path.join(os.getcwd(), name)
	if not os.path.exists(root_dir): os.mkdir(root_dir)

	import h5py
	file = h5py.File(os.path.join(root_dir, 'results.h5'), 'a')		# Create h5py file.

	if group_name: group = file.create_group(group_name) 			# Create group for multiple results with misalignment levels.
	else: group = file
	
	# Store metrics either in group or the dataset.
	group.create_dataset('rotation_error', data=np.array(r_errors))
	group.create_dataset('translation_error', data=np.array(t_errors))
	group.create_dataset('cd_loss', data=np.array(cd_loss))
	group.create_dataset('timings', data=np.array(timings))
	file.close()

	save_stats(t_errors, r_errors, cd_loss, timings, root_dir, group_name=group_name)
	
# Test a particular algorithm either with mask network or without it.
def test(args, model, reg_algorithm, test_loader):
	t_errors, r_errors, cd_loss, timings = test_one_epoch(args, model, reg_algorithm, test_loader)
	save_results(t_errors, r_errors, cd_loss, timings, args.results_dir, group_name=args.group_name)

def options():
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (Statistical Evaluation)')
	parser.add_argument('--dataset_path', type=str, default='testdata.h5',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')
	parser.add_argument('--kitti', type=bool, default=False, help='Train or Evaluate the network with KittiData.')

	# settings for on testing
	parser.add_argument('-b', '--batch_size', default=1, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	# algorithm for testing
	parser.add_argument('--masknet', default='True', type=str)
	parser.add_argument('--reg_algorithm', default='pointnetlk', type=str, choices=['pointnetlk', 'icp', 'dcp', 'prnet', 'pcrnet', 'rpmnet'])

	parser.add_argument('--pretrained_pnlk', default='../pretrained/exp_pointnetlk/models/no_noise_pointlk.pth', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--pretrained_dcp', default='../pretrained/exp_dcp/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--pretrained_prnet', default='../pretrained/exp_prnet/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--pretrained_pcrnet', default='../pretrained/exp_ipcrnet/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--pretrained_rpmnet', default='../pretrained/exp_rpmnet/models/partial-trained.pth', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--pretrained', default='../pretrained/model_masknet_ModelNet40.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')

	# results
	parser.add_argument('--results_dir', default='results_samplenet_pnlk', type=str,
						metavar='PATH', help='path to store results')
	parser.add_argument('--group_name', default='', type=str,
						metavar='PATH', help='path to store results')

	args = parser.parse_args()
	if args.masknet == 'False': args.masknet = False
	if args.masknet == 'True': args.masknet = True
	return args

def main():
	args = options()
	
	testset = TestDataLoader(args.dataset_path, args.group_name)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Define Registration Algorithm.
	if args.reg_algorithm == 'pointnetlk':
		pnlk = PointNetLK()
		if args.pretrained_pnlk:
			assert os.path.isfile(args.pretrained_pnlk)
			pnlk.load_state_dict(torch.load(args.pretrained_pnlk, map_location='cpu'))
			print("PointNetLK pretrained model loaded successfully!")
		pnlk = pnlk.to(args.device)
		reg_algorithm = pnlk

	elif args.reg_algorithm == 'icp':
		reg_algorithm = ICP()

	elif args.reg_algorithm == 'dcp':
		dgcnn = DGCNN(emb_dims=512)
		model = DCP(feature_model=dgcnn, cycle=True)
		model = model.to(args.device)
		if args.pretrained_dcp:
			assert os.path.isfile(args.pretrained_dcp)
			model.load_state_dict(torch.load(args.pretrained_dcp), strict=False)
			print("DCP pretrained model loaded successfully!")
		reg_algorithm = model

	elif args.reg_algorithm == 'prnet':
		model = PRNet()
		if args.pretrained_dcp:
			assert os.path.isfile(args.pretrained_prnet)
			model.load_state_dict(torch.load(args.pretrained_prnet, map_location='cpu'))
			print("PRNet pretrained model loaded successfully!")
		model = model.to(args.device)
		model.eval()
		reg_algorithm = model

	elif args.reg_algorithm == 'pcrnet':
		ptnet = PointNet(emb_dims=1024)
		model = iPCRNet(feature_model=ptnet)
		model = model.to(args.device)
		if args.pretrained_pcrnet:
			model.load_state_dict(torch.load(args.pretrained_pcrnet, map_location='cpu'))
			print("PCRNet pretrained model loaded successfully!")
		reg_algorithm = model

	elif args.reg_algorithm == 'rpmnet':
		model = RPMNet()
		model = model.to(args.device)
		if args.pretrained_rpmnet:
			model.load_state_dict(torch.load(args.pretrained_rpmnet, map_location='cpu')['state_dict'])
			print("RPMNet pretrained model loaded successfully!")
		reg_algorithm = model			

	# Define mask network.
	if args.masknet:
		model = MaskNet()
		model = model.to(args.device)
		if args.pretrained:
			assert os.path.isfile(args.pretrained)
			model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
			print("MaskNet pretrained model loaded successfully!")
		model.to(args.device)
	else:
		model = Identity()

	test(args, model, reg_algorithm, test_loader)

if __name__ == '__main__':
	main()