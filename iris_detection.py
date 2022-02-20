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


def daugman_scan(img, center, rad_min, rad_max, rad_step):
	pass



def detect_iris(img, rad_min=30, rad_max=150, rad_step=3, px_step=2, ratio=3/4):

	img_roi = ROI(img, ratio)
	cv2.imshow("ROI", img_roi)




# MAIN

db_path = "MMU-Iris-Database/"
nb_pers = 46
train_split = 4


train_set, test_set = import_db(db_path, nb_pers, train_split)

print("Taille train set :", len(train_set))
print("Taille test set :", len(test_set))


cv2.imshow(train_set[0][0], train_set[0][1])



# IRIS DETECTION

detect_iris(train_set[0][1])
cv2.waitKey(0)