import open3d as o3d
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import transforms3d.euler as t3d
import transforms3d
import time
	
from learning3d.models import PointNet, iPCRNet, RPMNet, PointNetLK
from learning3d.models import DGCNN, DCP, PRNet

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

# Specify Paths for Pretrained Models of Registration Networks.
def find_pretrained_path(reg_algorithm):
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	pretrained_reg = None
	if reg_algorithm == 'pointnetlk':
		pretrained_reg = 'pretrained/exp_pointnetlk/models/no_noise_pointlk.pth'
	elif reg_algorithm == 'dcp':
		pretrained_reg = 'pretrained/exp_dcp/models/best_model.t7'
	elif reg_algorithm == 'prnet':
		pretrained_reg = 'pretrained/exp_prnet/models/best_model.t7'
	elif reg_algorithm == 'pcrnet':
		pretrained_reg = 'pretrained/exp_ipcrnet/models/best_model.t7'
	elif reg_algorithm == 'rpmnet':
		pretrained_reg = 'pretrained/exp_rpmnet/models/partial-trained.pth'
	return os.path.join(BASE_DIR, pretrained_reg)


# Define Registration Algorithm.
def registration_algorithm(device=torch.device('cpu'), reg_algorithm='pointnetlk'):
	pretrained_reg = find_pretrained_path(reg_algorithm)

	if reg_algorithm == 'pointnetlk':
		pnlk = PointNetLK()
		if pretrained_reg:
			assert os.path.isfile(pretrained_reg)
			pnlk.load_state_dict(torch.load(pretrained_reg, map_location='cpu'))
			print("PointNetLK pretrained model loaded successfully!")
		pnlk = pnlk.to(device)
		reg_algorithm = pnlk

	elif reg_algorithm == 'icp':
		reg_algorithm = ICP()

	elif reg_algorithm == 'dcp':
		dgcnn = DGCNN(emb_dims=512)
		model = DCP(feature_model=dgcnn, cycle=True)
		model = model.to(device)
		if pretrained_reg:
			assert os.path.isfile(pretrained_reg)
			model.load_state_dict(torch.load(pretrained_reg, map_location='cpu'), strict=False)
			print("DCP pretrained model loaded successfully!")
		reg_algorithm = model

	elif reg_algorithm == 'prnet':
		model = PRNet()
		if pretrained_reg:
			assert os.path.isfile(pretrained_reg)
			model.load_state_dict(torch.load(pretrained_reg, map_location='cpu'))
			print("PRNet pretrained model loaded successfully!")
		model = model.to(device)
		model.eval()
		reg_algorithm = model

	elif reg_algorithm == 'pcrnet':
		ptnet = PointNet(emb_dims=1024)
		model = iPCRNet(feature_model=ptnet)
		model = model.to(device)
		if pretrained_reg:
			model.load_state_dict(torch.load(pretrained_reg, map_location='cpu'))
			print("PCRNet pretrained model loaded successfully!")
		reg_algorithm = model

	elif reg_algorithm == 'rpmnet':
		model = RPMNet()
		model = model.to(device)
		if pretrained_reg:
			model.load_state_dict(torch.load(pretrained_reg, map_location='cpu')['state_dict'])
			print("RPMNet pretrained model loaded successfully!")
		reg_algorithm = model

	return reg_algorithm


# Register template and source pairs.
class Registration:
	def __init__(self, reg_algorithm='pointnetlk'):
		self.reg_algorithm = reg_algorithm
		self.is_rpmnet = True if self.reg_algorithm == 'rpmnet' else False
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.reg_algorithm = registration_algorithm(device, reg_algorithm)

	@staticmethod
	def pc2points(data):
		if len(data.shape) == 3:
			return data[:, :, :3]
		elif len(data.shape) == 2:
			return data[:, :3]

	def register(self, template, source):
		# template, source: 		Point Cloud [B, N, 3] (torch tensor)

		# No need to use normals. Only use normals for RPM-Net.
		if not self.is_rpmnet == 'rpmnet':
			template, source = self.pc2points(template), self.pc2points(source)

		result = self.reg_algorithm(template, source)
		return result