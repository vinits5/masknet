import h5py
import numpy as np


def visualize_result(template, source, est_T):
	import open3d as o3d
	template, source, est_T = template, source, est_T
	transformed_source = np.matmul(est_T[0:3, 0:3], template.T).T + est_T[0:3, 3]

	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	transformed_source_ = o3d.geometry.PointCloud()
	
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
	
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	transformed_source_.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([template_, source_, transformed_source_])


class TestDataLoader:
	def __init__(self, name, group_name=''):
		file = h5py.File(name, 'r')
		if group_name: file = file[group_name]
		self.templates = np.array(file.get('templates'))
		self.sources = np.array(file.get('sources'))
		self.igt = np.array(file.get('igt'))
		self.masks = np.array(file.get('masks'))

	def __len__(self):
		return self.templates.shape[0]

	def __getitem__(self, idx):
		return self.templates[idx].astype(np.float32), self.sources[idx].astype(np.float32), self.igt[idx].astype(np.float32), self.masks[idx].astype(np.float32)


if __name__ == '__main__':
	test_set = TestDataLoader('testdata_various_angles.h5', 'angle_80')

	template, source, igt, _ = test_set[0]
	visualize_result(template, source, igt)