from segmentation import *
from normalisation import *
from feature_extraction_and_encoding import *
from smoothed_zscore_algo import *
from low_complexity import *

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
	