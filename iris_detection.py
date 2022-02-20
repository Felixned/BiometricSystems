import cv2, imutils
import numpy as np
import glob
import os




# DATA IMPORT

def import_db(db_path, nb_pers, train_split):


	path_left = "left/"
	path_right = "right/"

	train_set = []
	test_set = []

	for p in range(1, nb_pers+1):
		# Left
		path = db_path + "/" + str(p) + "/" + path_left + "*.bmp"
		img_left = [(os.path.basename(file), cv2.imread(file, cv2.IMREAD_GRAYSCALE)) for file in glob.glob(path)]

		train_set += img_left[:train_split]
		test_set += img_left[train_split:]

		# Right
		path = db_path + "/" + str(p) + "/" + path_right + "*.bmp"
		img_right = [(os.path.basename(file), cv2.imread(file, cv2.IMREAD_GRAYSCALE)) for file in glob.glob(path)]

		train_set += img_right[:train_split]
		test_set += img_right[train_split:]


	return train_set, test_set


# IRIS DETECTION with Daugman algorithm

def ROI(img, ratio):

	if ratio == 1.0: return img

	res = img.shape
	center = (res[0]//2, res[1]//2)
	img_roi = img[center[0]-int(res[0]*ratio/2):center[0]+int(res[0]*ratio/2), center[1]-int(res[1]*ratio/2):center[1]+int(res[1]*ratio/2)]

	return img_roi


def angle2dir(angle_matrix):
	# Correction
	angle_matrix = (angle_matrix + 180/8) % 360
	angle_matrix *= 8/360

	return angle_matrix.astype(int)

def non_max_suppr(amp, angle):
	amp_nonmax = np.zeros(amp.shape, dtype = np.uint8)
	angle_dir = angle2dir(angle)

	for y in range(1, amp_nonmax.shape[0]-1):
		for x in range(1, amp_nonmax.shape[1]-1):

			if angle_dir[y, x] in (0, 4):
				prev_amp = amp[y, x-1]
				next_amp = amp[y, x+1]

			elif angle_dir[y, x] in (1, 5):
				prev_amp = amp[y-1, x-1]
				next_amp = amp[y+1, x+1]

			elif angle_dir[y, x] in (2, 6):
				prev_amp = amp[y-1, x]
				next_amp = amp[y+1, x]

			elif angle_dir[y, x] in (3, 7):
				prev_amp = amp[y+1, x-1]
				next_amp = amp[y-1, x+1]


			if amp[y, x] >= prev_amp and amp[y, x] >= next_amp:
				amp_nonmax[y, x] = amp[y, x]

	return amp_nonmax


def amp_threshold(amp, low, high):

	amp_thresh = np.zeros(amp.shape, dtype = np.uint8)

	for y in range(amp.shape[0]):
		for x in range(amp.shape[1]):

			if amp[y, x] >= high : amp_thresh[y, x] = 255
			elif amp[y, x] <= low : amp_thresh[y, x] = 0
			else : amp_thresh[y, x] = 128

	return amp_thresh


def scan_strong(amp, y, x, ksize):
	strong = np.where(amp[y-ksize:y+ksize, x-ksize:x+ksize] == 255)
	return len(strong[0]) > 0


def hysteresis(amp, ksize):
	# Scan for strong neighbors edges

	rg = (ksize-1)//2

	amp_1 = amp.copy()
	for y in range(rg, amp.shape[0]-rg):
		for x in range(rg, amp.shape[1]-rg):

			if amp_1[y, x] == 128:
				amp_1[y, x] = (255 if scan_strong(amp_1, y, x, rg) else 0)

	amp_2 = amp.copy()
	for y in range(amp.shape[0]-rg, rg, -1):
		for x in range(amp.shape[1]):

			if amp_2[y, x] == 128:
				amp_2[y, x] = (255 if scan_strong(amp_2, y, x, rg) else 0)

	amp_3 = amp.copy()
	for y in range(rg, amp.shape[0]-rg):
		for x in range(amp.shape[1]-rg, rg, -1):

			if amp_3[y, x] == 128:
				amp_3[y, x] = (255 if scan_strong(amp_3, y, x, rg) else 0)

	amp_4 = amp.copy()
	for y in range(amp.shape[0]-rg, rg, -1):
		for x in range(amp.shape[1]-rg, rg, -1):

			if amp_4[y, x] == 128:
				amp_4[y, x] = (255 if scan_strong(amp_4, y, x, rg) else 0)


	final_amp = np.clip(amp_1 + amp_2 + amp_3 + amp_4, 0, 255)

	return final_amp



def weighted_canny(img, ksize, wx, visualize = True):
	
	# Denoising
	img_gaus = cv2.GaussianBlur(img, (ksize, ksize), 0)

	# Sobel
	grad_x = np.uint8(np.abs(cv2.Sobel(img_gaus, cv2.CV_64F, 1, 0, ksize)))
	grad_y = np.uint8(np.abs(cv2.Sobel(img_gaus, cv2.CV_64F, 0, 1, ksize)))

	amp = np.uint8(np.clip(2*np.sqrt((grad_x*wx)**2 + (grad_y*(1-wx))**2), 0, 255))
	angle = np.rad2deg(np.arctan2(grad_y, grad_x)) + 180


	# Non Max Suppression
	amp_nonmax = non_max_suppr(amp, angle)

	# 2 level thresholding
	amp_thresh = amp_threshold(amp_nonmax, 40, 80)
	
	# Hysteresis
	amp_hyst = hysteresis(amp_thresh, 7)
	

	if visualize:
		cv2.imshow("grad_x", grad_x)
		cv2.imshow("grad_y", grad_y)

		cv2.imshow("amp", amp)
		cv2.imshow("amp_nonmax", amp_nonmax)
		cv2.imshow("amp_thresh", amp_thresh)
		cv2.imshow("amp_hyst", amp_hyst)


	return amp_hyst



def daugman_scan(img, center, rad_min, rad_max, rad_step):
	pass



def detect_iris(img, rad_min=30, rad_max=150, rad_step=3, px_step=2, ratio=3/4):

	img_roi = ROI(img, ratio)
	cv2.imshow("ROI", img_roi)



# MAIN

db_path = "MMU-Iris-Database/"
nb_pers = 46
train_split = 4

print("Importing db...")
train_set, test_set = import_db(db_path, nb_pers, train_split)

print("Taille train set :", len(train_set))
print("Taille test set :", len(test_set))


img = train_set[0][1]

cv2.imshow(train_set[0][0], img)



# IRIS DETECTION

#detect_iris(img)
img_edge = weighted_canny(img, 3, 0.5)

cv2.waitKey(0)