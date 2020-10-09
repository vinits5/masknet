import os

def download_3dmatch():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'learning3d', 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	
	www = 'http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1.zip'
	zipfile = os.path.basename(www)
	www += ' --no-check-certificate'
	os.system('wget %s; unzip %s' % (www, zipfile))
	os.system('mv -r %s %s' % (zipfile[:-4], DATA_DIR))
	os.system('rm %s' % (zipfile))

if __name__ == '__main__':
	download_3dmatch()