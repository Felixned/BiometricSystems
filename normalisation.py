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
