import cv2, imutils
import numpy as np
import glob
import os
import time




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


# IRIS DETECTION with low complexity algorithm

image_height = 240
image_width = 320

#5 first algos are for pupil detection and the 6th one is for iris detection
def algo1_low_complexity(img):
	count = 0
	pupilX = 0
	pupilY = 0
	pupilDiameter = 0
	threshold = 0

	while pupilDiameter <= 0:
		#here we don't analyze all the pixels because sometime the first range of pixel is black and then the algorythm doesn't work (we assume that the center of the iris is not in the first or last 10 pixels of the images)
		for i in range(image_height//4, 3*image_height//4):
			difference = 0
			black_zone = False
			count = 0
			first_of_black_zone = 0
			last_of_black_zone = 0
			long_black_zone_first = 0
			long_black_zone_last = 0
			for j in range(image_width//4, 3*image_width//4):
				val = img[i][j]
				if val<threshold:
					count = count+1
					if black_zone:
						last_of_black_zone = j
					else:
						first_of_black_zone=j
						black_zone=True
					if count>=25:
						long_black_zone_first = first_of_black_zone;
						long_black_zone_last = last_of_black_zone;
				else:
					black_zone = False
					count = 0
			difference = long_black_zone_last - long_black_zone_first
			if difference>0:
				if pupilDiameter <= difference:
					pupilDiameter = difference 
					pupilX = (long_black_zone_last+long_black_zone_first)//2
					pupilY = i
		threshold = threshold + 20

	return img,pupilX, pupilY, pupilDiameter, threshold

def algo2_low_complexity(img, pupilX, pupilY, pupilDiameter, threshold):
	finished = False
	left = False
	right = False
	while finished == False:
		radius = pupilDiameter//2
		west_value = img[pupilY][pupilX-radius]
		east_value = img[pupilY][pupilX+radius]
		if east_value>threshold or west_value>threshold:
			if (east_value>threshold and west_value>threshold) or (left==True and right==True):
				pupilDiameter = pupilDiameter-2
				left = False
				right = False 
			else:
				if(west_value>threshold) and (east_value<=threshold):
					pupilX = pupilX+1
					right=True
				elif(east_value>threshold) and (west_value<=threshold):
					pupilX = pupilX-1
					left = True 
		else: 
			finished = True 

	return img,pupilX, pupilY, pupilDiameter, threshold

def algo3_low_complexity(img, pupilX, pupilY, pupilDiameter, threshold):
	finished = False
	up = False 
	down = False 
	while finished == False:
		radius = pupilDiameter//2
		north_value = img[pupilY-radius][pupilX]
		south_value = img[pupilY+radius][pupilX]
		if south_value>threshold or north_value>threshold:
			if (south_value>threshold and north_value>threshold) or (up==True and down==True) :
				pupilDiameter = pupilDiameter - 2
				up = False
				down = False
			else:
				if (north_value>threshold and south_value<=threshold):
					pupilY = pupilY+1
					down = True
				elif(south_value>threshold) and (north_value<=threshold):
					pupilY = pupilY-1
					up = True
		else:
			finished = True
	return img, pupilX, pupilY, pupilDiameter, threshold

def algo4_low_complexity(img, pupilX, pupilY, pupilDiameter, threshold):
	finished = False
	left = False
	right = False
	while finished == False:
		radius = pupilDiameter//2
		west_value = img[pupilY][pupilX-radius]
		east_value = img[pupilY][pupilX+radius]
		if east_value<=threshold or west_value<=threshold:
			if (east_value<=threshold and west_value<=threshold) or (left==True and right==True):
				pupilDiameter = pupilDiameter+2
				left = False
				right = False 
			else:
				if(west_value<=threshold) and (east_value>threshold):
					pupilX = pupilX-1
					left=True
				elif(east_value<=threshold) and (west_value>threshold):
					pupilX = pupilX+1
					right = True 
		else: 
			finished = True 

	return img,pupilX, pupilY, pupilDiameter, threshold



def algo5_low_complexity(img, pupilX, pupilY, pupilDiameter, threshold):
	finished = False
	up = False 
	down = False 
	while finished == False:
		radius = pupilDiameter//2
		north_value = img[pupilY-radius][pupilX]
		south_value = img[pupilY+radius][pupilX]
		if south_value<=threshold or north_value<=threshold:
			if (south_value<=threshold and north_value<=threshold) or (up==True and down==True) :
				pupilDiameter = pupilDiameter + 2
				up = False
				down = False
			else:
				if (north_value<=threshold and south_value>threshold):
					pupilY = pupilY-1
					down = True
				elif(south_value<=threshold) and (north_value>threshold):
					pupilY = pupilY+1
					up = True
		else:
			finished = True
	return img, pupilX, pupilY, pupilDiameter, threshold

#pas fini, de toute façon ça fonctionne même pas pour la pupille...

def algo6_low_complexity(img, pupilX, pupilY, pupilDiameter):
	irisL1X=0
	irisL1Y=0
	irisL1Diameter=0
	irisL2X=0
	irisL2Y=0
	irisL2Diameter=0
	#est-ce vrai ? pas précisé dans le papier... mdr
	value1,value2,value3,value4,value5,value6 = 0,0,0,0,0,0
	length = 8
	for k in range(-1,2):
		good = True
		finished = False
		irisY = pupilY + k
		x = pupilX - pupilDiameter//2
		irisDiameter = 0
		diameter = 0
		while True:
			for i in range(0,8):
				value1 = value1 + img[irisY][x-(length*0)-i]
				value2 = value2 + img[irisY][x-(length*1)-i]
				value3 = value3 + img[irisY][x-(length*2)-i]
				value4 = value4 + img[irisY][x-(length*3)-i]
				value5 = value5 + img[irisY][x-(length*4)-i]
				value6 = value6 + img[irisY][x-(length*5)-i]
			value1 = value1//length
			value2 = value2//length
			value3 = value3//length
			value4 = value4//length
			value5 = value5//length
			value6 = value6//length
			if value1<value4 and value1<value5 and value1<value6 and value2<value4 and value2<value5 and value2<value6 and value3<value4 and value3<value5 and value3<value6:
				length = i
				distLeft = ((pupilX-x)+3*length)+ length//2
				finished = True
			x = x-1
			if finished == True or (x-48)<0:
				break
		finished = False
		x = pupilX + pupilDiameter//2
		while True:
			for i in range(0,length):
				value1 = value1 + img[irisY][x+(length*0)+i]
				value2 = value2 + img[irisY][x+(length*1)+i]
				value3 = value3 + img[irisY][x+(length*2)+i]
				value4 = value4 + img[irisY][x+(length*3)+i]
				value5 = value5 + img[irisY][x+(length*4)+i]
				value6 = value6 + img[irisY][x+(length*5)+i]
			value1 = value1//length
			value2 = value2//length
			value3 = value3//length
			value4 = value4//length
			value5 = value5//length
			value6 = value6//length
			if value1<value4 and value1<value5 and value1<value6 and value2<value4 and value2<value5 and value2<value6 and value3<value4 and value3<value5 and value3<value6:
				distRight = ((x-pupilX)+3*length)+length//2
				finished = True
			x = x+1
			if finished == True or (x+48)>=image_width:
				break
		print(distLeft)
		print(distRight)
		if distLeft==0 and distRight>0 :
			distLeft = distRight
			good = False
		if distRight==0 and distLeft>0:
			distRight = distLeft
			good = False
		if 	distLeft>distRight and distLeft-distRight>18 :
			distRight=distLeft
			good = False
		if distRight>distLeft and distRight-distLeft>18:
			good = False
			distLeft=distRight
		if distLeft>distRight:
			irisX = pupilX-((distLeft-distRight)//2)
		else :
			irisX = pupilX+((distRight-distLeft)//2)
		irisDiameter = distLeft + distRight
		if good==True and (irisDiameter > irisL1Diameter):
			irisL1X = irisX
			irisL1Y = irisY
			irisL1Diameter = irisDiameter
		if good==False and (irisDiameter > irisL2Diameter):
			irisL2X = irisX
			irisL2Y = irisY
			irisL2Diameter = irisDiameter
	print(irisDiameter)
	if irisL1Diameter>0:
		return irisL1X, irisL1Y, irisL1Diameter
	elif irisL2Diameter>0:
		return irisL2X, irisL2Y, irisL2Diameter
	else:
		time.sleep(1)
		return None


def low_complexity(img, visualize = False):

	#blurred image in order to erase the reflections on the pupil
	blurred_img = cv2.medianBlur(img,25)


	#FIRST ALGO
	#print(algo1_low_complexity(blurred_img))
	result_algo1 = algo1_low_complexity(blurred_img)
	pupilX = result_algo1[1]
	pupilY = result_algo1[2]
	pupilDiameter = result_algo1[3]
	threshold = result_algo1[4]	

	"""
	print(pupilX)
	print(pupilY)
	print(pupilDiameter)
	print(threshold)
	"""

	img_algo1 = blurred_img.copy()
	img_algo1 = cv2.circle (img_algo1, (pupilX, pupilY), pupilDiameter//2, (255,255,255), 1)

	#SECOND ALGO
	#print(algo2_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold))
	result_algo2 = algo2_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold)
	pupilX = result_algo2[1]
	pupilY = result_algo2[2]
	pupilDiameter = result_algo2[3]
	threshold = result_algo2[4]	

	"""
	print(pupilX)
	print(pupilY)
	print(pupilDiameter)
	print(threshold)
	"""

	img_algo2 = blurred_img.copy()
	img_algo2 = cv2.circle (img_algo2, (pupilX, pupilY), pupilDiameter, (255,255,0), 1)

	#THIRD ALGO
	#print(algo3_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold))
	result_algo3 = algo3_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold)
	pupilX = result_algo3[1]
	pupilY = result_algo3[2]
	pupilDiameter = result_algo3[3]
	threshold = result_algo3[4]

	img_algo3 = blurred_img.copy()
	img_algo3 = cv2.circle (img_algo3, (pupilX, pupilY), pupilDiameter//2, (255,255,0), 1)

	#FOURTH ALGO
	#print(algo4_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold))
	result_algo4 = algo4_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold)
	pupilX = result_algo4[1]
	pupilY = result_algo4[2]
	pupilDiameter = result_algo4[3]
	threshold = result_algo4[4]

	img_algo4 = blurred_img.copy()
	img_algo4 = cv2.circle (img_algo4, (pupilX, pupilY), pupilDiameter//2, (255,255,0), 1)


	#FIFTH ALGO
	#print(algo5_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold))
	result_algo5 = algo5_low_complexity(blurred_img, pupilX, pupilY, pupilDiameter, threshold)
	pupilX = result_algo5[1]
	pupilY = result_algo5[2]
	pupilDiameter = result_algo5[3]
	threshold = result_algo5[4]

	img_algo5 = blurred_img.copy()
	img_algo5 = cv2.circle (img_algo5, (pupilX, pupilY), pupilDiameter//2, (255,255,0), 1)
	cv2.imshow("result pupil detection",img_algo5)

	#sixth algo
	
	result_algo6 = algo6_low_complexity(img, pupilX, pupilY, pupilDiameter//2)
	if algo6_low_complexity(img, pupilX, pupilY, pupilDiameter//2) != None:
		irisX = result_algo6[0]
		irisY = result_algo6[1]
		irisDiameter = result_algo6[2]
	img_algo6 = img_algo5.copy()
	img_algo6 = cv2.circle (img_algo5, (pupilX, pupilY), irisDiameter//2, (255,255,0), 1)
	cv2.imshow("result iris detection",img_algo6)

	if visualize:
		cv2.imshow("img", img)
		cv2.imshow("img_algo1",img_algo1)
		cv2.imshow("img_algo2",img_algo2)
		cv2.imshow("img_algo3",img_algo3)
		cv2.imshow("img_algo4",img_algo4)
		cv2.imshow("img_algo5",img_algo5)

	return img

# MAIN

db_path = "MMU-Iris-Database/"
nb_pers = 46
train_split = 4

print("Importing db...")
train_set, test_set = import_db(db_path, nb_pers, train_split)

print("Taille train set :", len(train_set))
print("Taille test set :", len(test_set))


img = train_set[4][1]
#cv2.imshow(train_set[0][0], img)


# IRIS DETECTION

pupil_edge = low_complexity(img, False)


cv2.waitKey(0)