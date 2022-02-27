import cv2, imutils
import numpy as np
import glob
import os, time

import matplotlib.pyplot as plt

import cProfile

from smoothed_zscore_algo import *


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
	
	# coo = np.where(amp_1[rg:amp.shape[0]-rg, rg:amp.shape[1]-rg] == 128)

	# amp_1[coo] = (255 if scan_strong(amp_1, coo[0], coo[1], rg) else 0)


	#amp_1[ amp_1[rg:amp.shape[0]-rg, rg:amp.shape[1]-rg] == 128 ] = 255

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



def weighted_canny(img, ksize, wx, low, high, visualize = True):
	"""
	img : grayscale img
	ksize : gaussian kernel
	wx : weight % on x sobel
	low : low threshold
	high : high threshold
	visualize = show all steps
	"""

	#https://www.adeveloperdiary.com/data-science/computer-vision/implement-canny-edge-detector-using-python-from-scratch/
	#https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
	
	# Denoising
	img_gaus = cv2.GaussianBlur(img, (ksize, ksize), 0)
	#img_gaus = cv2.GaussianBlur(img_gaus, (ksize, ksize), 0)

	# Sobel
	grad_x = np.uint8(np.abs(cv2.Sobel(img_gaus, cv2.CV_64F, 1, 0, ksize)))
	grad_y = np.uint8(np.abs(cv2.Sobel(img_gaus, cv2.CV_64F, 0, 1, ksize)))

	amp = np.uint8(np.clip(2*np.sqrt((grad_x*wx)**2 + (grad_y*(1-wx))**2), 0, 255))
	angle = np.rad2deg(np.arctan2(grad_y, grad_x)) + 180


	# Non Max Suppression
	amp_nonmax = non_max_suppr(amp, angle)

	# 2 level thresholding
	amp_thresh = amp_threshold(amp_nonmax, low, high)
	
	# Hysteresis
	amp_hyst = hysteresis(amp_thresh, 7) # odd value ksize
	

	if visualize:
		cv2.imshow("grad_x", grad_x)
		cv2.imshow("grad_y", grad_y)

		cv2.imshow("amp", amp)
		cv2.imshow("amp_nonmax", amp_nonmax)
		cv2.imshow("amp_thresh", amp_thresh)
		cv2.imshow("amp_hyst", amp_hyst)


	return amp_hyst


def test_bound(shape, y, x):
	if y < 0 or y > shape[0] :
		return False

	elif x < 0 or y > shape[1] :
		return False

	return True




def hough_circle(edge, accumulator, y, x, rad):

	list_deg = np.linspace(0, 360, int(2*np.pi*rad))
	list_y = np.clip(np.int32(np.round(np.sin(list_deg) * rad)) + y, 0, edge.shape[0]-1)
	list_x = np.clip(np.int32(np.round(np.cos(list_deg) * rad)) + x, 0, edge.shape[1]-1)

	accumulator[list_y, list_x] += 1


def hough_circle_cv(edge, accumulator, y, x, rad, color):
	cv2.circle(accumulator, (y, x), rad, color, 1)
	


def hough_transform(edge, rad_min=10, rad_max=200, rad_step=3, px_step=2, ratio=4/4):

	edge_roi = ROI(edge, ratio)
	cv2.imshow("ROI", edge_roi)

	nb_rad = 1+((rad_max - rad_min)//rad_step)

	accumulator = np.zeros((nb_rad, edge_roi.shape[0]+(2*rad_max), edge_roi.shape[1]+(2*rad_max)), dtype=np.uint32)
	temp = np.zeros(edge_roi.shape, dtype=np.uint8)


	nz_edges = np.nonzero(edge_roi)
	print("nz_edges", len(nz_edges[0]))

	for r in range(nb_rad):

		rad = rad_min + rad_step*r

		list_deg = np.arange(0, 360)
		list_y = np.int32(np.round(np.sin(list_deg) * rad))
		list_x = np.int32(np.round(np.cos(list_deg) * rad))

		for i in range(0, len(nz_edges[0]), px_step):

			y = nz_edges[0][i]
			x = nz_edges[1][i]

			# hough_circle_cv(edge, temp, y, x, rad, 1)
			# accumulator += temp
			# hough_circle_cv(edge, temp, y, x, rad, 0)

			#accumulator[np.clip(list_y+y, 0, edge_roi.shape[0]-1), np.clip(list_x+x, 0, edge_roi.shape[1]-1)] += 1
			accumulator[r, list_y+y+rad_max, list_x+x+rad_max] += 1


		print(rad)


	#print(accumulator)


	#cv2.imshow("accumulator", accumulator/np.max(accumulator))
	#max_acc_center = np.argmax(accumulator)

	accumulator_roi = accumulator[:, rad_max:rad_max+edge_roi.shape[0], rad_max:rad_max+edge_roi.shape[1]]


	return accumulator_roi


def hough_best_circles(accumulator, region, peak_ratio=2.5):

	plot_values = []
	plot_pnsr = []
	centers = []

	for r in range(accumulator.shape[0]):

		center_xy = np.unravel_index(np.argmax(accumulator[r, :, :], axis=None), accumulator[r, :, :].shape)

		#noise_region = accumulator[
								#np.clip(r-region, 0, img_hough.shape[0]) : np.clip(r+region, 0, img_hough.shape[0]),
								# r,
								# np.clip(center_xy[0]-region, 0, accumulator.shape[1]) : np.clip(center_xy[0]+region, 0, accumulator.shape[1]),
								# np.clip(center_xy[1]-region, 0, accumulator.shape[2]) : np.clip(center_xy[1]+region, 0, accumulator.shape[2])
								# ]

		center_value = accumulator[r, center_xy[0], center_xy[1]]

		#noise_value = (np.sum(noise_region)-center_value) / (noise_region.size-1)
		#psnr = center_value/noise_value

		#print("Radius index", r, "center", center_xy, "value", center_value, "psnr", np.round(psnr, 2), psnr>peak_ratio)

		plot_values.append(center_value)
		#plot_pnsr.append(psnr*3)
		centers.append(center_xy)


		img_rad = np.uint8(255*(accumulator[r, :, :]/np.max(accumulator[r, :, :])))

		cv2.circle(img_rad, (center_xy[1], center_xy[0]), 3, 255)

		cv2.imshow("Hough", img_rad)
		cv2.waitKey(1)


	return plot_values, plot_pnsr, centers



def peak_detect(values, lag=10, threshold=3.5, influence=0.1, visualize=False):
	results = thresholding_algo(values, lag=lag, threshold=threshold, influence=influence)

	peaks = results["signals"]
	avg = results["avgFilter"]
	std = results["stdFilter"]

	if visualize:
		plt.plot(range(len(values)), values)
		plt.plot(range(len(peaks)), peaks)

		plt.plot(range(len(avg)), avg)
		plt.plot(range(len(avg)), avg + threshold*std)
		plt.plot(range(len(avg)), avg - threshold*std)

		plt.show()
	#plt.plot(range(len(values)), psnr)

	peaks = np.transpose(np.argwhere(peaks == 1))[0].tolist()

	scored_peaks = []
	for peak in peaks:

		try:
			score = int(values[peak])-int(values[peak-1]) + int(values[peak])-int(values[peak+1])
		except Exception:
			score = 0

		scored_peaks.append([score, peak+1])

	return sorted(scored_peaks, reverse=True)



def normalization(img, center_pupille, radius_pupille, center_iris, radius_iris, shape):
	"""
	shape : (r_size, theta_size)
	"""

	seg_map = np.zeros(shape, dtype=np.uint8)

	#radius_delta = radius_iris - radius_pupille

	theta_range = np.linspace(0, 2*np.pi, shape[1])

	for t in range(shape[1]):
		theta = theta_range[t]

		ox = center_iris[0] - center_pupille[0]
		oy = center_iris[1] - center_pupille[1]
		alpha = ox**2 + oy**2
		beta = np.cos(np.pi - np.arctan2(oy, ox) - theta)

		r_max = radius_iris + np.sqrt(alpha)

		#print("ox oy alpha beta", ox, oy, alpha, beta)

		r_1 = np.sqrt(alpha)*beta + np.sqrt(-(alpha*(beta**2) - alpha - radius_iris**2))
		r_2 = np.sqrt(alpha)*beta - np.sqrt(-(alpha*(beta**2) - alpha - radius_iris**2))

		# print("r_1, r_2", r_1, r_2)

		if r_1 > r_2: r_ = r_1
		else : r_ = r_2

		#print("r_", r_)
		#print("max r_", r_max)



		xp = radius_pupille * np.cos(theta) + center_pupille[0]
		yp = radius_pupille * np.sin(theta) + center_pupille[1]
		xi = radius_iris * np.cos(theta) + center_iris[0]
		yi = radius_iris * np.sin(theta) + center_iris[1]


		r_range = np.linspace(radius_pupille, r_, shape[0])

		for i in range(shape[0]):
			r = r_range[i]

			r_ratio = r/r_

			x = r_ratio*(r_*np.cos(theta)) + center_pupille[0]
			y = r_ratio*(r_*np.sin(theta)) + center_pupille[1]

			# x = (r/r_max)*xi - (1- ((r-radius_pupille)/(r_max-radius_pupille)))*xp
			# y = (r/r_max)*yi - (1- ((r-radius_pupille)/r_max-radius_pupille))*yp

			# x = ((radius_iris-r) + radius_pupille)*xp + (r + radius_pupille)*xi
			# y = ((radius_iris-r) + radius_pupille)*yp + (r + radius_pupille)*yi


			# x = (radius_delta*(1-r) + radius_pupille)*xp + (radius_delta*r + radius_pupille)*xi
			# y = (radius_delta*(1-r) + radius_pupille)*yp + (radius_delta*r + radius_pupille)*yi

			#print(r, theta, y, x)

			seg_map[i, t] = img[int(x), int(y)]

	return seg_map



# MAIN

db_path = "MMU-Iris-Database/"
nb_pers = 46
train_split = 4

print("Importing db...")
train_set, test_set = import_db(db_path, nb_pers, train_split)

print("Taille train set :", len(train_set))
print("Taille test set :", len(test_set))


for i in range(0, 100):


	img = ROI(train_set[i][1], 3/4)

	cv2.imshow("Input", img)


	# IRIS DETECTION
	
	rad_step = 1
	lag = 10


	#detect_iris(img)

	t1 = time.time()

	img_edge = weighted_canny(img, ksize=5, wx=0.5, low=40, high=80, visualize=False)
	#img_edge = weighted_canny(img, 5, 0.5, 40, 80, False)

	cv2.imshow("Canny implementation", img_edge)


	"""
	Detect iris in rad_iris
	Detect pupille in iris roi
	"""

	# IRIS

	rad_iris = (35, 70)

	img_hough_iris = hough_transform(img_edge, rad_min=rad_iris[0], rad_max=rad_iris[1], rad_step=rad_step, px_step=2, ratio=4/4)
	values, psnr, centers = hough_best_circles(img_hough_iris, 20, 4)
	# scored_peaks = peak_detect(values, lag=lag, visualize=True)
	# print("Scored peaks", scored_peaks)

	best_index = np.argmax(values)
	center_iris = centers[best_index]

	radius_iris = rad_iris[0]+(best_index*rad_step)

	cv2.circle(img, (center_iris[1], center_iris[0]), radius_iris, 255)

	# for score, peak in scored_peaks[:1]:
	# 	cv2.circle(img, (centers[peak][1], centers[peak][0]), rad_iris[0]+(peak*rad_step), 255)


	cv2.imshow("Input", img)
	cv2.waitKey(1)




	# PUPILLE

	roi_ratio = 0.8

	rad_pupille = (10, int(radius_iris*0.8))

	iris_edge_roi = img_edge[center_iris[0]-int(radius_iris*roi_ratio):center_iris[0]+int(radius_iris*roi_ratio), center_iris[1]-int(radius_iris*roi_ratio):center_iris[1]+int(radius_iris*roi_ratio)]

	img_hough_pupille = hough_transform(iris_edge_roi, rad_min=rad_pupille[0], rad_max=rad_pupille[1], rad_step=rad_step, px_step=2, ratio=4/4)
	#cv2.waitKey(0)

	values, psnr, centers = hough_best_circles(img_hough_pupille, 20, 4)
	print(values)

	best_index = np.argmax(values)
	
	center_pupille = (centers[best_index][0] + center_iris[0] - int(radius_iris*roi_ratio), centers[best_index][1] + center_iris[1] - int(radius_iris*roi_ratio))
	radius_pupille = rad_pupille[0]+(best_index*rad_step)

	cv2.circle(img, (center_pupille[1], center_pupille[0]), radius_pupille, 255)


	scored_peaks = peak_detect(values, lag=lag, visualize=False)
	# print("Scored peaks", scored_peaks)


	# for score, peak in scored_peaks[:1]:
	# 	cv2.circle(img, (centers[peak][1], centers[peak][0]), rad_pupille[0]+(peak*rad_step), 255)

	cv2.imshow("Input", img)
	cv2.waitKey(1)


	##### RESULTS #####

	print("Iris :", center_iris, radius_iris, "Pupille :", center_pupille, radius_pupille)


	# CILS

	"""
	Hough linear transform on roi
	"""


	# NORMALIZATION

	seg_map = normalization(img, center_pupille, radius_pupille, center_iris, radius_iris, (80, 360))
	cv2.imshow("Seg", seg_map)

	print("Elapsed", time.time() - t1)
	cv2.waitKey(0)



	


#values_diff2 = np.diff(np.array(values).astype(np.int32))

#print(values_diff2)





# for i in range(len(img_hough)):

# 	cv2.imshow("Hough", img_hough[i, :, :]/np.max(img_hough[i, :, :]))

# 	center = np.unravel_index(np.argmax(img_hough[i, :, :], axis=None), img_hough[i, :, :].shape)
# 	print(i, center, img_hough[i, center[0], center[1]])

# 	img_rad = np.uint8(255*(img_hough[i, :, :]/np.max(img_hough[i, :, :])))

# 	cv2.circle(img_rad, (center[1], center[0]), 3, 255)

# 	cv2.imshow("Hough", img_rad)
# 	cv2.waitKey(0)


# blur = cv2.medianBlur(img,5)
# circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,param1=90,param2=10,minRadius=40,maxRadius=100)


# circles = circles.astype(int)[0]
# print(circles)

# for i in circles[0:1]:
# 	# draw the outer circle
# 	cv2.circle(img,(i[0],i[1]),i[2],255,1)
# 	# draw the center of the circle
# 	cv2.circle(img,(i[0],i[1]),2,255,3)


# cv2.imshow('detected circles',img)


# p = circles[0]
# roi = blur[p[1]-p[2]:p[1]+p[2], p[0]-p[2]:p[0]+p[2]]

# circles2 = cv2.HoughCircles(roi,cv2.HOUGH_GRADIENT,1,20,param1=90,param2=10,minRadius=10,maxRadius=40)
# circles2 = circles2.astype(int)[0]

# for i in circles2[0:1]:
# 	# draw the outer circle
# 	cv2.circle(img,(i[0]+p[0]-p[2],i[1]+p[1]-p[2]),i[2],255,1)
# 	# draw the center of the circle
# 	cv2.circle(img,(i[0]+p[0]-p[2],i[1]+p[1]-p[2]),2,255,3)


# cv2.imshow('detected circles',img)

#cv2.imshow("Canny cv2", cv2.Canny(img, 40, 80, 3))

