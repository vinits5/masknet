import open3d as o3d
import numpy as np 
import os
import copy
import time
import matplotlib.pyplot as plt

def create_pcd(xyz, color):
	# n x 3
	n = xyz.shape[0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	return pcd

def flip_geometries(pcds):
	pcds_transform = []
	flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
	for pcd in pcds:
		pcd_temp = copy.deepcopy(pcd)
		pcd_temp.transform(flip_transform)
		pcds_transform.append(pcd_temp)
	return pcds_transform

class Visualizer:
	def __init__(self, sleep_time=0.01):
		import time
		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window()
		self.sleep_time = sleep_time
		self.counter = 1

	def add_geometry(self, pcds):
		for pc in pcds: self.vis.add_geometry(pc)

	def remove_geometry(self, pcds, reset_bounding_box=False):
		for pc in pcds: self.vis.remove_geometry(pc, reset_bounding_box=reset_bounding_box)

	def update(self, pcds):
		for pc in pcds: self.vis.update_geometry(pc)

	def render(self, capture=False):
		self.set_zoom()
		self.vis.poll_events()
		self.vis.update_renderer()
		if capture: self.capture()
		time.sleep(self.sleep_time)

	def destroy(self):
		self.vis.destroy_window()

	def set_zoom(self):
		ctr = self.vis.get_view_control()
		ctr.set_zoom(1.1)

	def capture(self):
		image = self.vis.capture_screen_float_buffer(False)
		plt.imsave("images/{:03d}.png".format(self.counter), np.asarray(image), dpi = 30)
		self.counter += 1

	def rotate_view(self):
		ctr = self.vis.get_view_control()
		ctr.rotate(10.0, -0.0)
		

def display_results(template, source, est_T, mask_idx, est_T_series):
	non_mask_idx = np.array([i for i in range(1024) if i not in mask_idx])
	unmasked_template = template[non_mask_idx]
	masked_template = template[mask_idx]

	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]
	
	template_ = create_pcd(template, np.array([1, 0.706, 0]))
	source_ = create_pcd(source, np.array([0, 0.929, 0.651]))
	transformed_source_ = create_pcd(transformed_source, np.array([0, 0.651, 0.921]))
	masked_template_ = create_pcd(masked_template, np.array([1, 0.706, 0]))
	unmasked_template_ = create_pcd(unmasked_template, np.array([1,0,0]))

	template_, source_, transformed_source_, masked_template_, unmasked_template_ = flip_geometries([template_, source_, transformed_source_, masked_template_, unmasked_template_])

	vis = Visualizer()

	# Start creating initial_files (Contains template, source, masked_template and result of PointNetLK iterations)
	vis.add_geometry([template_, source_])
	vis.render(capture=True)

	vis.remove_geometry([template_])
	vis.add_geometry([masked_template_])
	vis.render(capture=True)
	
	transformed_source = create_pcd(source, np.array([0, 0.651, 0.921]))
	transformed_source = flip_geometries([transformed_source])[0]
	vis.add_geometry([transformed_source])
	vis.render(capture=True)

	for i in range(1, 11):
		est_T = est_T_series[i*4:(i+1)*4, :]
		transformed_source_i = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]
		transformed_source_i = create_pcd(transformed_source_i, np.array([0, 0.651, 0.921]))
		transformed_source_i = flip_geometries([transformed_source_i])[0]
		transformed_source.points = o3d.utility.Vector3dVector(transformed_source_i.points)
		vis.update([transformed_source])
		vis.render(capture=True)

	# Start creating files (Contains rotating view of [template, source and registered point cloud i.e. aligned with temnplate])
	vis.remove_geometry([masked_template_])
	vis.add_geometry([template_])
	vis.render(capture=True)

	for i in range(0, 220):
		vis.rotate_view()
		vis.render(capture=True)

	vis.destroy()

def read_data(path):
	data = np.load(path)
	template = data['template']
	source = data['source']
	est_T_series = data['est_T_series']
	est_T = data['est_T']
	mesh = data['mesh']
	mask_idx = data['mask_idx']
	return template, source, est_T, mask_idx, est_T_series


if __name__ == '__main__':
	path = '3dmatch_results.npz'
	template, source, est_T, mask_idx, est_T_series = read_data(path)

	if not os.path.exists('images'): os.mkdir('images')
	display_results(template, source, est_T, mask_idx, est_T_series)