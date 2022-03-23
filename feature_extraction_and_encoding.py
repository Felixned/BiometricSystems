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

