import cv2, imutils
import numpy as np
import glob
import os, time

import matplotlib.pyplot as plt

import cProfile

from smoothed_zscore_algo import *
from scipy.spatial.distance import hamming
from itertools import combinations

import pickle


# DATA IMPORT

def import_db(db_path, nb_pers, train_split):


	path_left = "left/"
	path_right = "right/"

	train_set = []
	test_set = []

	for p in range(1, nb_pers+1):
		# Left
		path = db_path + "/" + str(p) + "/" + path_left + "*.bmp"
		img_left = [(file, cv2.imread(file, cv2.IMREAD_GRAYSCALE)) for file in glob.glob(path)]

		train_set += img_left[:train_split]
		test_set += img_left[train_split:]

		# Right
		path = db_path + "/" + str(p) + "/" + path_right + "*.bmp"
		img_right = [(file, cv2.imread(file, cv2.IMREAD_GRAYSCALE)) for file in glob.glob(path)]

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
	# UNUSED
	if y < 0 or y > shape[0] :
		return False

	elif x < 0 or y > shape[1] :
		return False

	return True




def hough_circle(edge, accumulator, y, x, rad):
	# UNUSED
	list_deg = np.linspace(0, 360, int(2*np.pi*rad))
	list_y = np.clip(np.int32(np.round(np.sin(list_deg) * rad)) + y, 0, edge.shape[0]-1)
	list_x = np.clip(np.int32(np.round(np.cos(list_deg) * rad)) + x, 0, edge.shape[1]-1)

	accumulator[list_y, list_x] += 1


def hough_circle_cv(edge, accumulator, y, x, rad, color):
	# UNUSED
	cv2.circle(accumulator, (y, x), rad, color, 1)
	


def hough_transform_linear(edge, rho_num=0, theta_num = 360, px_step=2, ratio=4/4, visualize=False):

	edge_roi = ROI(edge, ratio)
	cv2.imshow("ROI", edge_roi)

	max_dist = np.sqrt(edge_roi.shape[0]**2 + edge_roi.shape[1]**2)
	if rho_num == 0:
		rho_num = int(max_dist)

	rho_res = max_dist/rho_num
	theta_res = 2*np.pi/theta_num

	accumulator = np.zeros((rho_num*2, theta_num), dtype=np.uint8) # positive & negative rho

	nz_edges = np.nonzero(edge_roi)


	theta_index_list = np.arange(0, theta_num)
	theta_list = theta_index_list * theta_res

	#rho_index_list = np.arange(0, rho_num)

	
	for i in range(0, len(nz_edges[0]), px_step):

		y = nz_edges[0][i]
		x = nz_edges[1][i]

		rho_list = (x*np.cos(theta_list) + y*np.sin(theta_list))/rho_res
		rho_index_list = rho_list.astype(int) + rho_num

		accumulator[rho_index_list, theta_index_list] += 1


	if visualize:
		cv2.imshow("Accumulator", accumulator[:rho_num, :]/np.max(accumulator[:rho_num, :]).astype(np.uint8))

	return accumulator[rho_num:, :], rho_res, theta_res


def houghline2line(rho_theta_list, img_shape):

	rho_list = rho_theta_list[:, 0]
	theta_list = rho_theta_list[:, 1]

	a = np.cos(theta_list)/np.sin(theta_list)
	b = rho_list/np.sin(theta_list)



	pt1_x = rho_list*np.cos(theta_list) + rho_list*np.sin(theta_list)*np.tan(theta_list)
	pt2_y = rho_list/np.sin(theta_list)

	print(a, b)



def hough_best_lines(accumulator, rho_res, theta_res, edge_shape, threshold=0.8):

	max_value = np.max(accumulator)

	best_centers_xy = np.argwhere(accumulator > max_value*threshold)
	best_values = accumulator[best_centers_xy[:, 0], best_centers_xy[:, 1]] #accumulator[best_centers_xy]

	for bc in best_centers_xy:
		cv2.circle(accumulator, (bc[1], bc[0]), 3, 255)
		print(bc, accumulator[bc[0], bc[1]])

	print(best_values)

	cv2.imshow("Accumulator", accumulator/max_value)



	# a = math.cos(best_centers_xy[:, 1])
	# b = math.sin(best_centers_xy[:, 1])
	# x0 = a * best_centers_xy[:, 0]
	# y0 = b * best_centers_xy[:, 0]
	# pt1 = (x0 + 1000*(-b)), int(y0 + 1000*(a))
	# pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
	# cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)




	best_rho_theta = best_centers_xy.copy().astype(float)
	best_rho_theta[:, 0] *= rho_res
	best_rho_theta[:, 1] *= theta_res

	lines = houghline2line(best_rho_theta, accumulator.shape)



	# center_xy = np.unravel_index(np.argmax(accumulator, axis=None), accumulator.shape)
	# center_value = accumulator[center_xy[0], center_xy[1]]

	#print(best_centers_xy, best_values)




def hough_transform_circle(edge, rad_min=10, rad_max=200, rad_step=3, px_step=2, ratio=4/4):

	edge_roi = ROI(edge, ratio)
	# cv2.imshow("ROI", edge_roi)

	nb_rad = 1+((rad_max - rad_min)//rad_step)

	accumulator = np.zeros((nb_rad, edge_roi.shape[0]+(2*rad_max), edge_roi.shape[1]+(2*rad_max)), dtype=np.uint32)
	#temp = np.zeros(edge_roi.shape, dtype=np.uint8)


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

		# cv2.imshow("Hough", img_rad)
		# cv2.waitKey(1)


	return plot_values, plot_pnsr, centers



def peak_detect(values, lag=10, threshold=3.5, influence=0.1, visualize=False):
	# UNUSED

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

	seg_map = np.zeros(shape, dtype=np.uint8) # output img

	#radius_delta = radius_iris - radius_pupille

	theta_range = np.linspace(0, 2*np.pi, shape[1])

	for t in range(shape[1]):
		theta = theta_range[t]

		ox = center_iris[0] - center_pupille[0]
		oy = center_iris[1] - center_pupille[1]
		alpha = ox**2 + oy**2 # distance²
		beta = np.cos(np.pi - np.arctan2(oy, ox) - theta) # angle

		#r_max = radius_iris + np.sqrt(alpha)

		#print("ox oy alpha beta", ox, oy, alpha, beta)

		# 2 solutions to equation
		r_1 = np.sqrt(alpha)*beta + np.sqrt(-(alpha*(beta**2) - alpha - radius_iris**2))
		r_2 = np.sqrt(alpha)*beta - np.sqrt(-(alpha*(beta**2) - alpha - radius_iris**2))

		# print("r_1, r_2", r_1, r_2)

		# Positive (maximum solution)
		if r_1 > r_2: r_ = r_1
		else : r_ = r_2

		#print("r_", r_)
		#print("max r_", r_max)


		# Radius positions at theta
		# xp = radius_pupille * np.cos(theta) + center_pupille[0]
		# yp = radius_pupille * np.sin(theta) + center_pupille[1]
		# xi = radius_iris * np.cos(theta) + center_iris[0]
		# yi = radius_iris * np.sin(theta) + center_iris[1]


		r_range = np.linspace(radius_pupille, r_, shape[0])

		for i in range(shape[0]):
			r = r_range[i]

			r_ratio = r/r_

			# Projection on x/y axis, center pupille origin
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

def remove_eyelashes(img):
	shape = img.shape
	for i in range(0,shape[0]):
		for j in range(0, shape[1]):
			if img[i][j] < 40:
				img[i][j]=0
	cv2.imshow("img removed_eyelashes",img)

def remove_light_reflections(img):
	shape = img.shape
	for i in range(0,shape[0]):
		for j in range(0, shape[1]):
			if img[i][j] > 180:
				img[i][j]=0
	cv2.imshow("img removed_eyelashes",img)

def remove_eyelids(img):
	return None



def G(f, f0, sigma):
	return np.exp(-(np.log(f/f0)**2) / (2*(np.log(sigma/f0))**2))


def feature_extraction(seg_map, filter, visualize=False):

	if seg_map == []:
		return []

	features = np.zeros((seg_map.size//2, 2), dtype=bool)

	for r in range(seg_map.shape[0]):

		signal = seg_map[r]

		x = np.arange(signal.shape[0])
		f = x[:signal.shape[0]]/signal.shape[0] # half frequency only
		signal_fft = np.fft.fft(signal) #[:len(f)] # half values only

		signal_filtered = signal_fft*filter(f)
		signal_filtered_ifft_real = np.fft.ifft(signal_filtered.real)[:len(f)//2]
		signal_filtered_ifft_imag = np.fft.ifft(signal_filtered.imag)[:len(f)//2]


		if visualize:
			#print(filter(f))
			#print(signal_filtered)
			gabor_ifft = np.fft.ifft(filter(f))
			plt.plot(f, filter(f))
			plt.show()
			plt.plot(f, signal_fft.real)
			plt.show()
			plt.plot(f, signal_filtered.real)
			plt.show()
			plt.plot(x[:signal.shape[0]//2], signal_filtered_ifft_imag)
			plt.show()


		features[signal_filtered_ifft_real.shape[0]*r:signal_filtered_ifft_real.shape[0]*(r+1), 0] = signal_filtered_ifft_real >= 0.0
		features[signal_filtered_ifft_real.shape[0]*r:signal_filtered_ifft_real.shape[0]*(r+1), 1] = signal_filtered_ifft_imag >= 0.0


	return features



def seg_eye(img_edge, rad_step, rad_iris=(35, 70), roi_ratio=0.8):

	"""
	Detect iris in rad_iris
	Detect pupille in iris roi
	"""

	# IRIS

	img_hough_iris = hough_transform_circle(img_edge, rad_min=rad_iris[0], rad_max=rad_iris[1], rad_step=rad_step, px_step=2, ratio=4/4)
	values, psnr, centers = hough_best_circles(img_hough_iris, 20, 4)
	# scored_peaks = peak_detect(values, lag=lag, visualize=True)
	# print("Scored peaks", scored_peaks)

	best_index = np.argmax(values)
	center_iris = centers[best_index]
	radius_iris = rad_iris[0]+(best_index*rad_step)


	# PUPILLE

	rad_pupille = (10, int(radius_iris*0.8))
	iris_edge_roi = img_edge[center_iris[0]-int(radius_iris*roi_ratio):center_iris[0]+int(radius_iris*roi_ratio), center_iris[1]-int(radius_iris*roi_ratio):center_iris[1]+int(radius_iris*roi_ratio)]

	img_hough_pupille = hough_transform_circle(iris_edge_roi, rad_min=rad_pupille[0], rad_max=rad_pupille[1], rad_step=rad_step, px_step=2, ratio=4/4)
	#cv2.waitKey(0)

	values, psnr, centers = hough_best_circles(img_hough_pupille, 20, 4)
	#print(values)

	best_index = np.argmax(values)
	center_pupille = (centers[best_index][0] + center_iris[0] - int(radius_iris*roi_ratio), centers[best_index][1] + center_iris[1] - int(radius_iris*roi_ratio))
	radius_pupille = rad_pupille[0]+(best_index*rad_step)



	#scored_peaks = peak_detect(values, lag=lag, visualize=False)
	# print("Scored peaks", scored_peaks)


	# for score, peak in scored_peaks[:1]:
	# 	cv2.circle(img, (centers[peak][1], centers[peak][0]), rad_pupille[0]+(peak*rad_step), 255)


	return center_pupille, radius_pupille, center_iris, radius_iris



def get_normalisation_eye(input_img, roi=3/4, rad_step=1, seg_shape=(20, 256), visualize=True):

	print("Beginning eye processing...")

	t1 = time.time()

	img = ROI(input_img, roi)
	if visualize:
		cv2.imshow("Input", img)

	# IRIS DETECTION
	#lag = 10

	print("Acquiring edges...")
	img_edge = weighted_canny(img, ksize=5, wx=0.5, low=40, high=80, visualize=False)
	if visualize:
		cv2.imshow("Canny implementation", img_edge)

	print("Segmentation...")
	center_pupille, radius_pupille, center_iris, radius_iris = seg_eye(img_edge, rad_step)
	print("Iris :", center_iris, radius_iris, "Pupille :", center_pupille, radius_pupille)

	if visualize:
		img_copy = img.copy()
		cv2.circle(img_copy, (center_iris[1], center_iris[0]), radius_iris, 255)
		cv2.circle(img_copy, (center_pupille[1], center_pupille[0]), radius_pupille, 255)
		cv2.imshow("Input", img_copy)
		cv2.waitKey(1)


	# EYELIDS
	"""
	Hough linear transform on iris_edge_roi
	"""

	# accu_lin, rho_res, theta_res = hough_transform_linear(img_edge, px_step=1, visualize=True)
	# hough_best_lines(accu_lin, rho_res, theta_res, img_edge.shape)
	# cv2.waitKey(0)


	
	# NORMALIZATION
	print("Normalization...")
	seg_map = normalization(img, center_pupille, radius_pupille, center_iris, radius_iris, seg_shape)
	if visualize:
		cv2.imshow("Seg", seg_map)
		cv2.waitKey(1)

	print("Elapsed", time.time() - t1, "\n")

	return seg_map



def get_features_eye(seg_map, lambda0=18, sigma_ratio=0.5):
	# Filter design

	# if seg_map != []:
	# 	f0 = lambda0/seg_map.shape[0]
	# else :
	# 	f0 = 0

	f0 = lambda0

	sigma = sigma_ratio*f0
	filter = lambda f: G(f, f0, sigma)

	#print("Feature extraction...")
	features = feature_extraction(seg_map, filter)
	features_line = np.ravel(features)

	#print("Nb features:", features_line.size)

	#print(features_line)


	return features_line



def feature_matching(features1, features2, shifts=3, size_shift=2):
	# return minium distance for all shifts

	if features1.size == 0 or features2.size == 0:
		return None

	dist_list = []

	for i in range(-shifts, shifts+1):
		dist_list.append(hamming(features1, np.roll(features2, size_shift*i)))

	#print(dist_list)

	return min(dist_list)



def decidability(mean1, mean2, std1, std2):
	return np.abs(mean2 - mean1) / np.sqrt((std2**2 + std1**2)/2)


def gen_inter_intra(features_set, visualize=True):

	matching_list_intra = []
	matching_list_inter = []

	for i, j in combinations(range(len(train_norm_set)), 2):

		features1 = features_set[i][1]
		features2 = features_set[j][1]

		hamming_score = feature_matching(features1, features2)

		if hamming_score != None:
			if j < ((i//4)+1)*4: # intra class
				matching_list_intra.append(hamming_score)
			else :
				matching_list_inter.append(hamming_score)


		#print(i)

	print("[INTRA] Mean", np.mean(matching_list_intra), "Std", np.std(matching_list_intra))
	print("[INTER] Mean", np.mean(matching_list_inter), "Std", np.std(matching_list_inter))

	deci = decidability(np.mean(matching_list_intra), np.mean(matching_list_inter), np.std(matching_list_intra), np.std(matching_list_inter))
	print("Decidability =", deci)



	far = len(np.where(np.array(matching_list_inter) <= separation_point))/len(matching_list_inter)
	frr = len(np.where(np.array(matching_list_intra) >= separation_point))/len(matching_list_intra)

	print("FAR =", far, "FRR =", frr)


	if visualize:
		plt.hist(matching_list_inter, bins=100, density=True) #, density=True
		plt.hist(matching_list_intra, bins=100, density=True)
		plt.show()

	return deci


def gen_decidability_plot():

	decidability_plot = []
	#range_plot = range(1, 70, 3) # lambda0
	range_plot = np.arange(0.1, 0.9, 0.1) #sigma


	for x in range_plot:

		print("Getting features for x =", x)

		# GET FEATURES
		#print("Getting features from", len(train_norm_set), "seg maps")
		features_set = []

		for norm in train_norm_set:
			name, seg_map = norm

			features = get_features_eye(seg_map, 1/43, x)

			features_set.append([name, features])

		#print("features_set len", len(features_set))


		decidability_plot.append(gen_inter_intra(features_set, visualize=False))


	plt.scatter(list(range_plot), decidability_plot)
	plt.show()





# MAIN

db_path = "MMU-Iris-Database/"
nb_pers = 46
train_split = 4

print("Importing db...")
train_set, test_set = import_db(db_path, nb_pers, train_split)

print("Taille train set :", len(train_set))
print("Taille test set :", len(test_set))

# NORMALISATION SAVE

# for i in range(len(train_set)):

# 	input_img_name = train_set[i][0].split(".")[0]
# 	print(input_img_name)
# 	store_name = input_img_name + "_seg.pck"

# 	input_img = train_set[i][1]

# 	try:
# 		seg_map = get_normalisation_eye(input_img, visualize=True)
# 	except Exception:
# 		seg_map = []

# 	f = open(store_name, 'wb')
# 	pickle.dump(seg_map, f)
# 	f.close()


# NORMALISATION IMPORT

train_norm_set = []

for i in range(len(train_set)):

	input_img_name = train_set[i][0].split(".")[0]
	store_name = input_img_name + "_seg.pck"

	try:
		f = open(store_name, 'rb')
		seg_map = pickle.load(f)
		f.close()

	except Exception:
		print("File not found", store_name)


	train_norm_set.append([train_set[i][0], seg_map])

#print(train_norm_set)


# GET FEATURES

print("Getting features from", len(train_norm_set), "seg maps")
features_set = []

for norm in train_norm_set:
	name, seg_map = norm

	features = get_features_eye(seg_map, 1/42, 0.4)

	features_set.append([name, features])

print("features_set len", len(features_set))

print("Matching...")



# PROCESS
separation_point = 0.36

#gen_inter_intra(features_set, visualize=True)
#gen_decidability_plot()


"""
# INTER - INTRA plot

matching_list_intra = []
matching_list_inter = []

for i, j in combinations(range(len(train_norm_set)), 2):

		features1 = features_set[i][1]
		features2 = features_set[j][1]

		hamming_score = feature_matching(features1, features2)

		if hamming_score != None:
			if j < ((i//4)+1)*4: # intra class
				matching_list_intra.append(hamming_score)
			else :
				matching_list_inter.append(hamming_score)


	#print(i)

print("[INTRA] Mean", np.mean(matching_list_intra), "Std", np.std(matching_list_intra))
print("[INTER] Mean", np.mean(matching_list_inter), "Std", np.std(matching_list_inter))

deci =  decidability(np.mean(matching_list_intra), np.mean(matching_list_inter), np.std(matching_list_intra), np.std(matching_list_inter))
print("Decidability =", deci)



far = len(np.where(np.array(matching_list_inter) <= separation_point))/len(matching_list_inter)
frr = len(np.where(np.array(matching_list_intra) >= separation_point))/len(matching_list_intra)

print("FAR =", far, "FRR =", frr)


#decidability_plot.append(deci)


#plt.scatter(list(range_plot), decidability_plot)
#plt.show()

plt.hist(matching_list_inter, bins=100, density=True)
plt.hist(matching_list_intra, bins=100, density=True)


plt.show()


"""


success = 0

# TEST INPUT

#time_list = []

#for input_img_index in range(len(test_set)):

#t1 = time.time()

input_img_index = int(input("Eye index > "))
input_img = test_set[input_img_index][1]

#try:
seg_map = get_normalisation_eye(input_img, visualize=True)
#except Exception:
#	continue
features_base = get_features_eye(seg_map, 1/42, 0.4)

#TEST SCAN ALL TRAIN_SET
matching_list = []

for i in range(len(train_norm_set)):

	#seg_map = train_norm_set[i][1]
	features = features_set[i][1]

	hamming_score = feature_matching(features_base, features)

	#print("Matching:", hamming_score, "\n")
	if hamming_score != None:
		matching_list.append(hamming_score)
	else :
		matching_list.append(1.0)


print(np.argwhere(np.array(matching_list) <= separation_point))
print("Awaited results", input_img_index*4, "to", (input_img_index+1)*4 - 1)
print("Best result", np.argmin(np.array(matching_list)))

if np.argmin(np.array(matching_list)) >= input_img_index*4 and np.argmin(np.array(matching_list)) < (input_img_index+1)*4 :
	print("SUCCESS !")
	success += 1

#time_list.append(time.time() - t1)

print("")

print("Nb success", success)
print("Success rate", success/len(test_set))


plt.scatter(list(range(len(matching_list))), matching_list)
plt.show()

#print("TIME Mean", np.mean(time_list), "Std", np.std(time_list))

#plt.hist(time_list, bins=10)
#plt.show()







# TEST HAMMING

# input_img = train_set[0][1]
# features1 = get_features_eye(input_img, visualize=True)

# input_img = train_set[6][1]
# features2 = get_features_eye(input_img, visualize=True)

# print("Matching:", feature_matching(features1, features2))
	



	


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

