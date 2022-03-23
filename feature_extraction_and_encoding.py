import numpy as np
from scipy.spatial.distance import hamming


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