import cv2, imutils
import numpy as np

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