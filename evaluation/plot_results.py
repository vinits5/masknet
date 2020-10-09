import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Class to read results from a h5py file.
class ReadResults:
	def __init__(self, file_path, various_data=True):
		self.file = h5py.File(os.path.join(file_path, 'results.h5'), 'r')
		self.various_data = various_data 									# True if file_path has results of misalignment levels or noise levels.

	@staticmethod
	def get_keys(file):
		return [key for key in file.keys()]

	# Read keys of data and groups in the dataset.
	def read_file_keys(self):
		if self.various_data:
			self.data_variation_keys = self.get_keys(self.file) 						# Keys for angles/noises like [angle_0, angle_10, etc.]
			self.metric_keys = self.get_keys(self.file[self.data_variation_keys[0]])	# Keys for metrices like [cd, rot_err, trans_err, etc.]		
		else:
			self.metric_keys = get_keys(self.file)
			self.data_variation_keys = None

	# Read data in given group and metric key.
	def read_file_data(self, group, metric_key):
		if group is None:
			return np.array(self.file.get(metric_key))
		else:
			grp = self.file[group]
			return np.array(grp.get(metric_key))

	def __call__(self, mean=True):
		self.read_file_keys() 			# First read all the keys.
		results = {}

		# If contains groups.
		if self.data_variation_keys is not None:
			for dvk in self.data_variation_keys:		# Loop over all groups.
				results[dvk] = {}
				for mk in self.metric_keys:				# Loop over all keys for each group.
					if mean: 
						results[dvk][mk] = np.mean(self.read_file_data(dvk, mk))		# collect mean of stored data.
					else:
						results[dvk][mk] = self.read_file_data(dvk, mk)					# collect the entire data.

		# If doesn't contain groups.
		else:
			for mk in self.metric_keys:			# Loop over all keys.
				if mean:
					results[mk] = np.mean(self.read_file_data(self.data_variation_keys, mk))		# collect mean of stored data.
				else:
					results[mk] = self.read_file_data(self.data_variation_keys, mk)					# collect the entire data.

		return results


# Read data from given list of files.
def read_all_files(files):
	results = [ReadResults(f)() for f in files]
	return results

# Define range of angles to be tested.
def angle_range():
	return np.arange(0, 60, 10)

# Define range of noises to be tested.
def noise_range():
	return [0, 0.02, 0.04, 0.06]#, 0.1, 0.15, 0.2]

# Define range of outliers to be tested.
def outlier_range():
	return np.arange(0, 100, 10).tolist()+ np.arange(100, 1100, 100).tolist()

# List mean of all misalignment levels for a specific metric value.
def get_angle_data(results, metric_key):
	data = []
	for angle in angle_range():
		key = 'angle_'+str(angle)
		data.append(results[key][metric_key])
	return data

# List mean of all noise levels for a specific metric value.
def get_noise_data(results, metric_key):
	data = []
	for noise in noise_range():
		key = 'noise_'+str(noise)
		data.append(results[key][metric_key])
	return data

# List mean of all outlier levels for a specific metric value.
def get_outliers_data(results, metric_key):
	data = []
	for outliers in outlier_range():
		key = 'outliers_'+str(outliers)
		data.append(results[key][metric_key])
	return data


class Plotter:
	def __init__(self, root_dir, files, labels, colors, case='angle', metric='rotation_error'):
		self.case = case
		self.metric = metric
		self.files = files
		self.root_dir = root_dir
		self.labels = labels
		self.colors = colors

		self.files_full_path = [os.path.join(root_dir, f) for f in self.files]
		self.results = read_all_files(self.files_full_path)

		if case == 'noise': 
			self.get_data = get_noise_data
			self.x_values = noise_range()
		elif case == 'angle': 
			self.get_data = get_angle_data
			self.x_values = angle_range()
		elif case == 'outlier':
			self.get_data = get_outliers_data
			self.x_values = [x*(100/1000) for x in outlier_range()] 		# Get percentage of outliers.

		fig = plt.figure()
		self.ax = plt.subplot(111)

	# Create a plot.
	def make_single_plot(self, y_values, idx):
		if 'samplenet' in self.files[idx]: linestyle = '-'
		else: linestyle = '--'
		self.ax.plot(self.x_values, y_values, linestyle=linestyle, linewidth=4, label=self.labels[idx], color=self.colors[idx])
		self.ax.scatter(self.x_values, y_values, s=45, c=self.colors[idx])

	# Define plot settings.
	def plot_settings(self):
		xlabel, ylabel = self.get_labels()

		plt.xlabel(xlabel, fontsize=35)
		plt.ylabel(ylabel, fontsize=35)
		plt.tick_params(labelsize=30, width=3, length=10)
		
		if self.case == 'angle': 
			plt.xticks(np.arange(0,50.5,10))
			plt.xlim(-0.5,50.5)
		elif self.case == 'noise':
			plt.xticks(np.arange(0,0.0605,0.01))
			plt.xlim(-0, 0.0605)
		elif self.case == 'outlier':
			plt.xticks(np.arange(0, 100.5, 10))
			plt.xlim(-0, 100.5)
		
		plt.grid(True)
		self.set_label_pose()

	def set_label_pose(self):
		box = self.ax.get_position()
		self.ax.set_position([box.x0, box.y0 + box.height * 0.2,
		                 box.width, box.height * 0.9])

		# Put a legend below current axis
		self.ax.legend(loc=9, bbox_to_anchor=(0.5, -0.15),
		          fancybox=True, shadow=True, ncol=10, fontsize=30)

	def get_labels(self):
		if self.case == 'angle':
			if self.metric == 'rotation_error': x_label, y_label = 'Initial Angle (in Deg.)', 'Rotation Error (Deg.)'
			elif self.metric == 'translation_error': x_label, y_label = 'Initial Angle (in Deg.)', 'Translation Error'

		if self.case == 'noise':
			if self.metric == 'rotation_error': x_label, y_label = 'Noise (Std. Dev.)', 'Rotation Error (Deg.)'
			elif self.metric == 'translation_error': x_label, y_label = 'Noise (Std. Dev.)', 'Translation Error'

		if self.case == 'outlier':
			if self.metric == 'rotation_error': x_label, y_label = 'Percentage of outliers', 'Rotation Error (Deg.)'
			elif self.metric == 'translation_error': x_label, y_label = 'Percentage of outliers', 'Translation Error'

		return x_label, y_label

	def create_plot(self):
		for idx, res in enumerate(self.results):
			y_values = self.get_data(res, self.metric)
			self.make_single_plot(y_values, idx)
		self.plot_settings()
		
		plt.show()

def find_files(args):
	if args.case == 'angle': ending = 'rotations'
	elif args.case == 'noise': ending = 'noises'
	elif args.case == 'outlier': ending = 'outliers'
	X_reg = 'results_'+args.reg_algorithm+'_various_'+ending
	mask_X_reg = 'results_masknet_'+args.reg_algorithm+'_various_'+ending
	return [X_reg, mask_X_reg]

def find_colors(args):
	if args.reg_algorithm == 'icp': return ['C0', 'C0']
	elif args.reg_algorithm == 'pointnetlk': return ['C1', 'C1']
	elif args.reg_algorithm == 'dcp': return ['C2', 'C2']
	elif args.reg_algorithm == 'prnet': return ['C3', 'C3']
	elif args.reg_algorithm == 'rpmnet': return ['C4', 'C4']
	else: print('Registration algorithm is not defined! Please check the code.')

def find_labels(args):
	if args.reg_algorithm == 'icp': label = 'ICP'
	elif args.reg_algorithm == 'pointnetlk': label = 'PointNetLK'
	elif args.reg_algorithm == 'dcp': label = 'DCP'
	elif args.reg_algorithm == 'prnet': label = 'PRNet'
	elif args.reg_algorithm == 'rpmnet': label = 'RPMNet'
	if label:
		return [label, 'Mask-'+label]
	else: print('Registration algorithm is not defined! Please check the code.')


def options():
	parser = argparse.ArgumentParser(description='Plot Results of Mask-X vs X registration algorithms')
	parser.add_argument('--root_dir', type=str, default='', help='Specify location of stored results')
	parser.add_argument('--case', type=str, default='angle', choices=['angle', 'noise', 'outlier'],
						help='Give value as per the testing')
	parser.add_argument('--metric', type=str, default='rotation_error', 
						choices=['rotation_error', 'translation_error'],
						help='Decide metric to plot on y-axis')
	parser.add_argument('--reg_algorithm', type=str, default='pointnetlk', 
						choices=['pointnetlk', 'dcp', 'icp', 'prnet', 'rpmnet'],
						help='Choose registration algorithm (X) in the plot.')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = options()

	files = find_files(args)
	labels = find_labels(args)
	colors = find_colors(args)
	
	plotter = Plotter(args.root_dir, files, labels, colors, case=args.case, metric=args.metric)
	plotter.create_plot()