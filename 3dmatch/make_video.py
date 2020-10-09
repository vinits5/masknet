import os
from tqdm import tqdm
import cv2

def get_files(root_dir):
	files = os.listdir(root_dir)
	files.sort()
	return files[:14], files[14:]

def selectROI(im):
	r = cv2.selectROI(im)
	x, y, h, w = r[0], r[1], r[3], r[2]
	return x, y, h, w

def write(crop_img, video, desired_fps, original_fps):
	iterations = int(original_fps/desired_fps)
	if iterations == 1: 
		video.write(crop_img)
	else:
		for i in range(iterations):
			video.write(crop_img)

def make_video(root_dir, initial_files, files, setting):
	print("Making the video!")
	x, y, h, w = setting	
	size = (w, h)
	fps = 30
	zoom_fps = 5
	slow_fps = 1
	last_fps = 1
	video = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, size)

	for idx, file in enumerate(tqdm(initial_files)):
		img = cv2.imread(os.path.join(root_dir, file))
		crop_img = img[y:y+h, x:x+w]

		if idx in [0, 1, len(initial_files)-1]: write(crop_img, video, slow_fps, fps)
		else: write(crop_img, video, zoom_fps, fps)

	for idx, file in enumerate(tqdm(files)):
		img = cv2.imread(os.path.join(root_dir, file))
		crop_img = img[y:y+h, x:x+w]

		if idx == 0: write(crop_img, video, slow_fps, fps)
		if idx == len(files)-1: write(crop_img, video, last_fps, fps)
		else: write(crop_img, video, fps, fps)

	video.release()
	print("Video finished")

if __name__ == '__main__':
	root_dir = 'images'
	initial_files, files = get_files(root_dir)

	setting = selectROI(cv2.imread(os.path.join(root_dir, initial_files[0])))
	make_video(root_dir, initial_files, files, setting)