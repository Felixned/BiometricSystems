import time
import cv2, imutils
import numpy as np
from segmentation import *



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


	matching_list_intra = []
	matching_list_inter = []

	for i, j in combinations(range(len(train_norm_set)), 2):

		features1 = features_set[i][1]
		features2 = features_set[j][1]

		hamming_score = feature_matching(features1, features2)



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
