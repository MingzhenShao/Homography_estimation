import tensorflow as tf 
import numpy as np 
import os, os.path, datetime
import cv2, re, random, csv
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from glob import glob
from numpy.linalg import inv

import scipy.ndimage
from scipy import stats
from multiprocessing import Process, Queue


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

run_id = 'baseline'
log_dir = './H_predict/log/{}'.format(run_id)
ckpt_dir = './H_predict/ckpt/{}'.format(run_id)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
import logging


######## Pre-method ############

def homography_regression_model(input_img):
	
	x = tf.layers.conv2d(inputs=input_img, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)

	# without fc layer (dense)	
	# It's hard to analyze how fc layers works.
	
	x = tf.layers.conv2d(inputs=x, filters=8, strides=1, kernel_size=3, padding="same", activation=None)
	x1 = x
	out = tf.reduce_mean(x, axis=[1,2])

	'''
	x = tf.layers.flatten(x)
	#x = tf.layers.dropout(x, rate=0.75)
	x = tf.layers.dense(x, 1024)
	x1 = x
	out = tf.layers.dense(x, 8)
	'''
	return out, x1

#################### ResNet #################################

def res_block(feature_in, out_channel, stride):
	current = tf.layers.conv2d(inputs=feature_in, filters=out_channel, strides=stride, kernel_size=1, padding="same")
	current = tf.layers.batch_normalization(inputs=current)
	current = tf.nn.relu(current)

	current = tf.layers.conv2d(inputs=current, filters=out_channel, strides=1, kernel_size=3, padding="same")
	current = tf.layers.batch_normalization(inputs=current)
	current = tf.nn.relu(current)

	current = tf.layers.conv2d(inputs=current, filters=out_channel*4, strides=1, kernel_size=1, padding="same")
	current = tf.layers.batch_normalization(inputs=current)

	return current


def homography_regression_resnet(img):
#The input shape (N, 224, 224, 3)

	tmp = tf.layers.conv2d(inputs=img, filters=64, strides=2, kernel_size=7, padding="same")
	tmp = tf.layers.batch_normalization(inputs=tmp)
	tmp = tf.nn.relu(tmp)
	#tmp = tf.nn.pool()		couldn't fix it
	tmp = tf.nn.max_pool(value=tmp, ksize=[1,3,3,1], padding="SAME", strides=[1,2,2,1])

	res_bock_out = res_block(tmp, 64, 1)
	tmp_1 = tf.layers.conv2d(inputs=tmp, filters=256, strides=1, kernel_size=1, padding="same")
	tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
	tmp = tf.add(res_bock_out, tmp_1)
	tmp = tf.nn.relu(tmp)

	for i in range(2):
		res_bock_out = res_block(tmp, 64, 1)
		tmp = tf.add(res_bock_out, tmp)
		tmp = tf.nn.relu(tmp)

	res_bock_out = res_block(tmp, 128, 2)
	tmp_1 = tf.layers.conv2d(inputs=tmp, filters=512, strides=2, kernel_size=1, padding="same")
	tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
	tmp = tf.add(res_bock_out, tmp_1)
	tmp = tf.nn.relu(tmp)

	for i in range(3):
		res_bock_out = res_block(tmp, 128, 1)
		tmp = tf.add(res_bock_out, tmp)
		tmp = tf.nn.relu(tmp)

	res_bock_out = res_block(tmp, 256, 2)
	tmp_1 = tf.layers.conv2d(inputs=tmp, filters=1024, strides=2, kernel_size=1, padding="SAME")
	tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
	tmp = tf.add(res_bock_out, tmp_1)
	tmp = tf.nn.relu(tmp)

	for i in range(5):

		res_bock_out = res_block(tmp, 256, 1)
		tmp = tf.add(res_bock_out, tmp)
		tmp = tf.nn.relu(tmp)

	res_bock_out = res_block(tmp, 512, 2)
	tmp_1 = tf.layers.conv2d(inputs=tmp, filters=2048, strides=2, kernel_size=1, padding="same")
	tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
	tmp = tf.add(res_bock_out, tmp_1)
	tmp = tf.nn.relu(tmp)

	for i in range(2):
		res_bock_out = res_block(tmp, 512, 1)
		tmp = tf.add(res_bock_out, tmp)
		tmp = tf.nn.relu(tmp)
	# out = tf.nn.avg_pool(value=tmp, ksize=[1,7,7,1], padding="SAME", strides=[1,7,7,1])

	x = tf.layers.conv2d(inputs=tmp, filters=8, strides=1, kernel_size=3, padding="same", activation=None)
	out = tf.reduce_mean(x, axis=[1,2])
		
	return out, x


#################### VGG16 #################################

def homography_regression_vgg(img):
#The input shape (N, 224, 224, 3)

	x = tf.layers.conv2d(inputs=img, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=64, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	# x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=128, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	# x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=256, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=256, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=256, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	# x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	# x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)


	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	# x = tf.layers.batch_normalization(inputs=x)
	x = tf.layers.max_pooling2d(x, 2, 2)

	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.conv2d(inputs=x, filters=512, strides=1, kernel_size=3, padding="same", activation=tf.nn.relu)
	x = tf.layers.batch_normalization(inputs=x)
	# x = tf.layers.flatten(x)
	# # x = tf.layers.dropout(x, rate=0.75)
	# x = tf.layers.dense(x, 4096)
	# x = tf.layers.dense(x, 4096)
	# out = tf.layers.dense(x, 8)


	x = tf.layers.conv2d(inputs=x, filters=8, strides=1, kernel_size=3, padding="same", activation=None)
	out = tf.reduce_mean(x, axis=[1,2])

	return out, x

#############################################################
# # function for training and test
# def get_train(path = "~/Documents/DATASET/VOC2011/train/*.jpg", num_examples = 1280):
# 	# hyperparameters
# 	rho = 32
# 	patch_size = 224
# 	height = 320
# 	width = 320

# 	loc_list = glob(path)
# 	X = np.zeros((num_examples,128, 128, 2))  # images
# 	Y = np.zeros((num_examples,8))
# 	for i in range(num_examples):
# 		# select random image from tiny training set
# 		index = random.randint(0, len(loc_list)-1)
# 		img_file_location = loc_list[index]
# 		color_image = plt.imread(img_file_location)
# 		color_image = cv2.resize(color_image, (width, height))
# 		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# 		# create random point P within appropriate bounds
# 		y = random.randint(rho, height - rho - patch_size)  # row?
# 		x = random.randint(rho, width - rho - patch_size)  # col?
# 		# define corners of image patch
# 		top_left_point = (x, y)
# 		bottom_left_point = (patch_size + x, y)
# 		bottom_right_point = (patch_size + x, patch_size + y)
# 		top_right_point = (x, patch_size + y)
# 		four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
# 		perturbed_four_points = []
# 		for point in four_points:
# 			perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

# 		# compute H
# 		H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
# 		H_inverse = inv(H)
# 		inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 240))
# 		warped_image = cv2.warpPerspective(gray_image, H, (320, 240))

# 		# grab image patches
# 		original_patch = gray_image[y:y + patch_size, x:x + patch_size]
# 		warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
# 		# make into dataset
# 		training_image = np.dstack((original_patch, warped_patch))
# 		H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
# 		X[i, :, :] = training_image
# 		Y[i, :] = H_four_points.reshape(-1)		
# 	return X,Y


# def pre_deform(path, num_examples = 256):
# 	loc_list = glob(path)
# 	X = []
# 	Y = []
# 	for i in range(num_examples):
# 		index = random.randint(0, len(loc_list)-1)
# 		img_file_location = loc_list[index]
# 		color_image = cv2.imread(img_file_location)
			
# 		try:
# 			gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
# 		except Exception as e:
# 			print(img_file_location)
# 			continue
# 		ori, deform = deformation.Deform(gray_image, 10, img_size=512)
# 		X.append(ori)
# 		Y.append(deform)
# 	return X, Y

def get_generator(queue, path, num_examples = 256):

	for k in range(1000):
		# ori_data, deform_data = pre_deform(path, num_examples*4)
		for k in range(120):
	#	while 1:
			# hyperparameters
			rho = 32
			patch_size = 224
			height = 320
			width = 320

			loc_list = glob(path)

			X = np.zeros((num_examples,224, 224, 2))  # images
			Y = np.zeros((num_examples,8))
			for i in range(num_examples):
				# select random image from tiny training set
				index = random.randint(0, len(loc_list)-1)

				# index = random.randint(0, len(ori_data)-1)
				# ori = ori_data[index]
				# deform = deform_data[index]
				# ori = cv2.resize(ori, (width, height))
				# deform = cv2.resize(deform, (width, height))

				#### White Noise Image ###
				# img_data = []
				# pixel_color = ""
				
				# gray_image = Image.new("L", size=(320, 320))
				# for k in range(320*320):
				# 	img_data.append(random.randint(0, 255))
				# gray_image.putdata(img_data)
				# gray_image = np.array(gray_image)

				############################

				img_file_location = loc_list[index]
				color_image = cv2.imread(img_file_location)
				
				try:
					gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
				except Exception as e:
					print(img_file_location)
					continue
				gray_image = cv2.resize(gray_image, (width, height))
				
				# create random point P within appropriate bounds
				y = random.randint(rho, height - rho - patch_size)  # row
				x = random.randint(rho, width - rho - patch_size)  # col
				# define corners of image patch
				top_left_point = (x, y)
				bottom_left_point = (patch_size + x, y)
				bottom_right_point = (patch_size + x, patch_size + y)
				top_right_point = (x, patch_size + y)
				four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
				perturbed_four_points = []
				for point in four_points:
					perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

				# compute H
				H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
				H_inverse = inv(H)
				inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (320, 320))
				
				# grab image patches
				original_patch = gray_image[y:y + patch_size, x:x + patch_size]
				warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]

				# make into dataset
				training_image = np.dstack((original_patch, warped_patch))
				H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))


				X[i, :, :] = training_image
				Y[i, :] = H_four_points.reshape(-1)		
			queue.put((np.array(X), np.array(Y)), block = True)
	#	yield (X,Y)
	return

def get_test(path):

	rho = 32
	patch_size = 224
	height = 320
	width = 320
	# #random read image
	# loc_list = glob(path)
	# index = random.randint(0, len(loc_list)-1)
	# img_file_location = loc_list[index]

	# #For *.png
	# if(img_file_location.split('.')[-1] == 'png'):
	# 	color_image = np.array(cv2.imread(img_file_location))		# Why use np.array(Image.open(img_file_location)) for *.png ?
	# else:
	# color_image = cv2.imread(img_file_location)
	color_image = cv2.imread(path)

	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(path)
		# print(img_file_location)
	
	gray_image = cv2.resize(gray_image, (width, height))

	# ##### Change color while keep texture
	# gray_image = 255 - gray_image

	#points
	######deformation image#######
	# ori, deform = deformation.Deform(gray_image, 10, img_size=512)
	# ori = cv2.resize(ori, (width, height))
	# deform = cv2.resize(deform, (width, height))
	# ori = np.float32(ori)/255
	# deform = np.float32(deform)/255

	### White Noise Image ###

	# img_data = []
	# gray_image = Image.new("L", size=(320, 320))
	# for i in range(320*320):
	# 	img_data.append(random.randint(0, 255))
	# gray_image.putdata(img_data)
	# gray_image = np.array(gray_image)


	y = random.randint(rho, height - rho - patch_size)  # row
	x = random.randint(rho,  width - rho - patch_size)  # col
	top_left_point = (x, y)
	bottom_left_point = (patch_size + x, y)
	bottom_right_point = (patch_size + x, patch_size + y)
	top_right_point = (x, patch_size + y)
	four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	four_points_array = np.array(four_points)
	perturbed_four_points = []
	for point in four_points:
		perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

	print(perturbed_four_points)
	print(four_points_array)
		
	#compute H
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	
	# make into dataset
	training_image = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))
	H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
	
	# print(index, x, y, random.randint(-rho, rho), random.randint(-rho, rho))

	return training_image, H_four_points.reshape(-1), np.array(four_points).reshape(-1), color_image, gray_image, inv_warped_image
	


def get_test_phy(path):

	rho = 32
	patch_size = 224
	height = 320
	width = 320

	img_dir_1 = os.path.join(path, "1.ppm")
	img_dir_2 = os.path.join(path, "2.ppm")
	matrix_dir = os.path.join(path, "H_1_2")
	
	_fm = open(matrix_dir)
	_m = _fm.readlines()
	
	color_image = cv2.imread(img_dir_1)
	color_image_warped = cv2.imread(img_dir_2)

	
	# scale_x = 2
	# scale_y = 2
	# print(scale_x, scale_y, color_image.shape[0], color_image.shape[1])
	scale = 2


	M = np.array([list(map(float, s.strip().split())) for s in _m])
	# print(H_matrix)
	# H_matrix = np.array([[_matrix[0,0]*scale_x, _matrix[0,1]*scale_y, _matrix[0,2]*scale_x], [_matrix[1,0]*scale_x, _matrix[1,1]*scale_y, _matrix[1,2]*scale_y], [_matrix[2,0], _matrix[2,1], _matrix[2,2]]])
	H_matrix = np.array([[M[0,0]*scale, M[0,1], M[0,2]*scale], [M[1,0], M[1,1]*scale, M[1,2]*scale], [M[2,0], M[2,1], M[2,2]]])


	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
		gray_image_warped = cv2.cvtColor(color_image_warped, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(path)
		# print(img_file_location)
	
	gray_image = gray_image[100:100+640, 100:100+640]
	gray_image_warped = gray_image_warped[100:100+640, 100:100+640]

	gray_image = cv2.resize(gray_image, (width, height))
	gray_image_warped = cv2.resize(gray_image_warped, (width, height))


	y = 36
	x = 42
	# top_left_point = (x*scale, y*scale)
	# bottom_left_point = ((patch_size + x)*scale, y*scale)
	# bottom_right_point = ((patch_size + x)*scale, (patch_size + y)*scale)
	# top_right_point = (x*scale, (patch_size + y)*scale)

	top_left_point = (x, y)
	bottom_left_point = ((patch_size + x), y)
	bottom_right_point = ((patch_size + x), (patch_size + y))
	top_right_point = (x, (patch_size + y))


	z_value = 1
	init_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	points_np = np.array(init_points).astype(np.float32) 
	points_3d = np.column_stack((points_np, np.full((len(points_np), 1), z_value)))

	print(points_3d)

	perturbed_four_points_list = []
	for point in points_3d:
	    # Perform matrix multiplication
	    transformed_point = np.matmul(point, H_matrix)
	    
	    # Convert back to Cartesian coordinates
	    # transformed_point /= transformed_point[2]
	    
	    # Append the transformed point to the list
	    perturbed_four_points_list.append(transformed_point[:2])

	perturbed_four_points = np.array(perturbed_four_points_list).astype(np.int32)

	four_points_array = np.array([top_left_point, bottom_left_point, bottom_right_point, top_right_point]).astype(np.int32)


	
	inv_warped_image = gray_image_warped
	# cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	
	# make into dataset
	training_image = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))
	H_four_points = np.subtract(perturbed_four_points, four_points_array)
	print(H_four_points, four_points_array)
	print(inv_warped_image.shape)
	
	return training_image, H_four_points.reshape(-1), four_points_array.reshape(-1), color_image, gray_image, inv_warped_image
	

def get_test_demo(path, rand_list):
	rho = 32
	patch_size = 224
	height = 320
	width = 320
	#random read image
	loc_list = glob(path)
	index = rand_list[0]
	img_file_location = loc_list[index]

	#For *.png
	if(img_file_location.split('.')[-1] == 'png'):
		color_image = np.array(Image.open(img_file_location))
	else:
		color_image = cv2.imread(img_file_location)

	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(img_file_location)

	gray_image = cv2.resize(gray_image,(width,height))

	#points
	y = rand_list[1]  # row
	x = rand_list[2]  # col
	top_left_point = (x, y)
	bottom_left_point = (patch_size + x, y)
	bottom_right_point = (patch_size + x, patch_size + y)
	top_right_point = (x, patch_size + y)
	four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	four_points_array = np.array(four_points)
	perturbed_four_points = []
	for point in four_points:
		perturbed_four_points.append((point[0] + rand_list[3], point[1] + rand_list[4]))
		
	#compute H
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	# make into dataset
	training_image = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))
	H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
	
	return training_image, H_four_points.reshape(-1), np.array(four_points).reshape(-1), color_image
	
	# return color_image, H_inverse,val_image,four_points_array

def get_test_visualization(loc_list, index, H_four_points, four_points):

	H_four_points = H_four_points.reshape([4,2])
	four_points = four_points.reshape([4,2])
	rho = 32
	patch_size = 224
	height = 320
	width = 320
	# #random read image
	# loc_list = glob(path)
	# index = random.randint(0, len(loc_list)-1)
	img_file_location = loc_list[index]

	try:
		color_image = cv2.imread(img_file_location)
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(img_file_location)
	
	gray_image = cv2.resize(gray_image, (width, height))

	y = four_points[0][0]   # row
	x = four_points[0][1] # col
	
	perturbed_four_points = four_points + H_four_points
	
	#compute H
	#print(four_points, perturbed_four_points)
	H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	H_inverse = inv(H)
	inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
	# grab image patches
	original_patch = gray_image[y:y + patch_size, x:x + patch_size]
	warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
	
	# make into dataset
	val_images = np.dstack((original_patch, warped_patch))
	# val_image = training_image.reshape((1,224,224,2))

	# print(index, x, y, random.randint(-rho, rho), random.randint(-rho, rho))

	return val_images, gray_image, inv_warped_image, original_patch, warped_patch

def consumer(queue):
	X, Y = queue.get(block = True)
	return X, Y


def result_show_diff(name, gray_image, inv_warped_image, pred_val, H_points, base_points):		# This method leads to a resolution decrease. 

	print(name, pred_val, H_points, base_points)
	#quit()
	
	perturbed_four_points = base_points + pred_val 		#use H_points to show it's right
	H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points.reshape([4,2])), np.float32(base_points.reshape([4,2])))
	H_inverse = inv(H)
	predict_image = cv2.warpPerspective(inv_warped_image, H_inverse, (320, 320))
	diff_img = predict_image - gray_image

	plt.imshow(predict_image)
	plt.savefig('{}pred.png'.format(name))
	plt.imshow(gray_image)
	plt.savefig('{}gray.png'.format(name))
	plt.imshow(diff_img)
	plt.colorbar()
	plt.savefig('{}diff.png'.format(name))
	
	plt.close('all')



def result_show_rectangle(gray_image, inv_warped_image, original_patch, warped_patch, pred_val, H_points, base_points):
	pass

	transform_img = cv2.warpPerspective(cv2.resize(ori_img,(320,320)), H_inverse, (320, 320))
	# transform_img = deform_img 
	transform_img_ = transform_img.copy()
	color_image = ori_img 

	ax=plt.subplot(2,3,1)
	# plt.title("ori_img")
	color_image_draw = cv2.polylines(color_image, np.array([base_points.reshape((4,2))], np.int32), 1, (255,0,0), 2)
	ax.imshow(color_image_draw, vmin=0, vmax=255, cmap='jet')

	
	label_4point = (-H_points + base_points).reshape((4,2))	
	pred_4point = np.int32(np.reshape((-pred + base_points),(4,2)))
	# print(pred_4point, label_4point)

	cv2.polylines(transform_img, np.array([label_4point], np.int32), 1, (255,0,0), 2)
	# cv2.polylines(transform_img, np.array([pred_4point], np.int32), 1, (0,0,255), 2)
	# cv2.polylines(transform_img, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
	# mce = np.sum(abs(pred-H_points))/8
	# print(mce)

	# cv2.putText(transform_img,'Mean Corner Error = {0:.2f}'.format(mce),(20,20), 4, 0.5, (255,0,0),1,cv2.LINE_AA)

	ax=plt.subplot(2,3,2)
	# plt.title("transform_img")
	ax.imshow(transform_img, vmin=0, vmax=255, cmap='jet')

	ax=plt.subplot(2,3,3)

	cv2.polylines(transform_img_, np.array([base_points.reshape((4,2))], np.int32), 1, (0,0,255), 2)
	# plt.title("transform_img")
	ax.imshow(transform_img_, vmin=0, vmax=255, cmap='jet')

	# plt.subplots_adjust(wspace=-0.35)
	plt.savefig('{}.png'.format(item))
	
	plt.show()

def batch_norm_featuremap(pred_val, feature_map):
	pass
	pred_val_idx = []
	feature_map_idx = []
	pred_val_idx.append(pred_val)
	feature_map_idx.append(feature_map)

	max_feature = np.amax(feature_map_idx)
	min_feature = np.min(feature_map_idx)
	feature_range = max_feature - min_feature

	norm_feature = (feature_map-min_feature)/feature_range


def save_fig(pred_val, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, feature_map, H_points, base_points, boardline_feature, feature_max=None, feature_min=None):

	name = 'S_' + str(np.int(np.mean(np.abs(pred_val - H_points)))) + '_S_'
	_feature_map = np.around(np.array(feature_map), 2)

	for e in pred_val:
		name += str(e)
		name += '_'
	sub_img, ax = plt.subplots(2,2, tight_layout=True)

	# Show image pairs	
	for e in ax.reshape(-1):
		e.axis('off')
	ax[0,0].imshow(gray_image, cmap='gray')
	ax[0,1].imshow(inv_warped_image, cmap='gray')
	ax[1,0].imshow(original_patch, cmap='gray')
	ax[1,1].imshow(warped_patch, cmap='gray')
	plt.savefig(save_dir+'/{}img.png'.format(name))
	plt.close('all')

	# # Show feature maps
	
	fig_f, ax_f = plt.subplots(2,4, tight_layout=True)
	idx = 0
	for e in ax_f.reshape(-1):
		e.axis('off')
		e.imshow(feature_map[0,:,:,idx], vmin=feature_min, vmax=feature_max, cmap='gray')
		idx += 1

	# # colorbar() ?!

	plt.savefig(save_dir+'/{}feature.png'.format(name))
	plt.close('all')


	# # Give the hist of each channel

	# fig_m, ax_m = plt.subplots(2,4, tight_layout=True, figsize=(8,4))
	# idx = 0
	# for e in ax_m.reshape(-1):
	# 	e.hist(_feature_map[0,:,:,idx])
	# 	idx += 1
	# plt.savefig(save_dir+'/{}matrix.png'.format(name))
	# plt.close('all')

	###################################
	# Show focus on the input image for each channel
	###################################

	fig_F, ax_F = plt.subplots(2,4, sharex='all', sharey='all', tight_layout=True, figsize=(12,6))			
	idx = 0
	
	sub_ave_list = []
	for e in ax_F.reshape(-1):
		# if(abs(pred_val[idx]-H_points[idx])>5):				# if the error is too large, do NOT visualize this feature.
			# bin_feature_map = None
			# e.text(0.2, 0.2, 'Loss error', ha='center', va='center', fontsize=28, color='C1')
			# idx += 1
			# print("skip!")
			# continue
		bin_feature_map = _feature_map[0,:,:,idx].copy()

		sub_feature_map = bin_feature_map - boardline_feature[0,:,:,idx]
		sub_feature_map = np.around(sub_feature_map, 0)

		_sub_feature_map = np.zeros(np.shape(sub_feature_map))
		mask = (sub_feature_map != 0)

		np.putmask(_sub_feature_map, mask, bin_feature_map)

		if(np.count_nonzero(_sub_feature_map) != 0):
			sub_ave = np.sum(_sub_feature_map) / np.count_nonzero(_sub_feature_map)
			sub_ave_list.append(sub_ave)
		
		mask_feature_map = scipy.ndimage.zoom(abs(sub_feature_map), 8, order=1)
		
		# Visualize the position in original image.
		# mask = bin_feature_map.reshape(np.shape(original_patch))

		# extent = np.min(original_patch), np.max(original_patch), np.min(mask_feature_map), np.max(mask_feature_map)		# Here is the problem!!!

		# e.set(adjustable='box', aspect='equal')
		input_pathches = original_patch + warped_patch
		im1 = e.imshow(input_pathches, cmap=plt.cm.gray)
		im2 = e.imshow(mask_feature_map, cmap='bwr', alpha=.4)
	
		idx += 1

	plt.subplots_adjust(wspace=-0.15, hspace=-0.15)
	# print(sub_ave_list)
	# if(len(sub_ave_list) == 8):
	# 	print("sus_ave: {}".format(str(np.around(sub_ave_list - H_points))))
	# 	print("prev: {}".format(str(np.around(pred_val - H_points))))

	plt.savefig(save_dir+'/{}hist_map.png'.format(name))
	plt.close('all')
	

	# # Save the matrix
	f_mat = open(save_dir+'/{}matrix.csv'.format(name), 'w')
	feature_map_str = _feature_map.astype(str)

	csv_f = csv.writer(f_mat)
	for i in range(np.shape(feature_map)[-1]):
		for j in range(np.shape(feature_map)[1]):
			csv_f.writerow(feature_map_str[0,j,:,i])
		csv_f.writerow('')

	f_mat.close()

	if(len(sub_ave_list) == 8):
		sub_ave_mean = np.mean(abs(sub_ave_list - H_points))

		return sub_ave_mean
	else:
		return 0


def val_err(val_dir):
	f = open(val_dir + "_color_ori_ERR_VGG.txt", 'w')

	sum_err =0
	counter = 0
	i=1
	# print(os.path.join('.',val_dir, 'v_'))

	for item in glob(os.path.join(val_dir, 'v_*')):
		# print(item)
		f.write(str(item)+ '||: ')

		image_pair, H_points, base_points, color_image, ori_img, deform_img = get_test_phy(item)
		
		f.write(str(H_points)+ '  >>>  ')

		pred, boardline_feature = sess.run([logits, x1], feed_dict={datas: image_pair.reshape([1,224,224,2])})
		f.write(str(pred.astype(int).reshape(-1)) + '  >>>  ')

		# _ = save_fig(pred.astype(int).reshape(-1), './', ori_img, deform_img, ori_img, deform_img, boardline_feature, H_points, base_points, boardline_feature)

		H_abs = H_points + base_points
		deform_img = cv2.cvtColor(deform_img, cv2.COLOR_GRAY2RGB)

		try: 
			deform_img = cv2.line(deform_img, (H_abs[0], H_abs[1]), (H_abs[2], H_abs[3]), (255,0,0), 3)
			deform_img = cv2.line(deform_img, (H_abs[2], H_abs[3]), (H_abs[4], H_abs[5]), (255,0,0), 3)
			deform_img = cv2.line(deform_img, (H_abs[4], H_abs[5]), (H_abs[6], H_abs[7]), (255,0,0), 3)
			deform_img = cv2.line(deform_img, (H_abs[6], H_abs[7]), (H_abs[0], H_abs[1]), (255,0,0), 3)
			
			print(H_abs[0], H_abs[1], H_abs[2], H_abs[3], H_abs[4], H_abs[5], H_abs[6], H_abs[7])
			quit()
			
			pred_abs = pred.astype(int).flatten() + base_points
			# print(H_abs, pred_abs)
			deform_img = cv2.line(deform_img, (pred_abs[0], pred_abs[1]), (pred_abs[2], pred_abs[3]), (0,0,255), 3)
			deform_img = cv2.line(deform_img, (pred_abs[2], pred_abs[3]), (pred_abs[4], pred_abs[5]), (0,0,255), 3)
			deform_img = cv2.line(deform_img, (pred_abs[4], pred_abs[5]), (pred_abs[6], pred_abs[7]), (0,0,255), 3)
			deform_img = cv2.line(deform_img, (pred_abs[6], pred_abs[7]), (pred_abs[0], pred_abs[1]), (0,0,255), 3)
		except Exception as e:
			print(e)

		cv2.imwrite("{}_1.jpg".format(i), ori_img)

		cv2.imwrite("{}.jpg".format(i), deform_img)
		i+=1


		err = np.mean(np.abs(pred-H_points))
		f.write('ERR: ' + str(err) + '  |||\r')

		sum_err += err 
		counter += 1
	print("ERR: {}".format((1.0*sum_err)/counter))



def val_dis(val_dir):

	recorder = open("recorder.txt", 'w')
	for item in glob(os.path.join(val_dir, '*_pre.jpg')):
		print(item)
		recorder.write(str(item)+ '||: ')

		undis_image, H_points, base_points, color_image, ori_img, deform_img = get_test(item)
		recorder.write(str(H_points)+ '  >>>  ')

		save_dir = os.path.join(val_dir, 'outputImg')
		if(not os.path.exists(save_dir)):
			os.makedirs(save_dir)
		

		pred_ori, boardline_feature = sess.run([logits, x1], feed_dict={datas: undis_image.reshape([1,224,224,2])})
		# reference_dir = os.path.join(save_dir, 'reference')
		# os.makedirs(reference_dir)
		# _ = save_fig(pred_ori.astype(int).reshape(-1), save_dir, ori_img, deform_img, ori_img, deform_img, boardline_feature, H_points, base_points, boardline_feature)

		recorder.write(str(pred_ori.astype(int).reshape(-1)) + '  >>>  ')
			
		########################

		item_dis = item.split('_pre.jpg')[0] + '_dis.jpg'
		# print(' ========= {} ========= '.format(item))
		err = 0
		# err_sub_ave_mean = 0
		# cnt_err = 0
		
		loc_list = glob(item_dis)
		# cnt_img = len(loc_list) -1

		# for i in range(cnt_img):	
		dis_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, 0, H_points, base_points)

		pred_dis, xx = sess.run([logits, x1], feed_dict={datas: dis_images.reshape([1,224,224,2])})
		#print(np.shape(xx))
		pred_dis = np.around(np.array(pred_dis), 2).reshape(-1)
		pred_int_dis = pred_dis.astype(int)
		recorder.write(str(pred_int_dis) + '  >>>  ')
		
		# sub_ave_mean = save_fig(pred_int_dis, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)


		################################
		# # Channel normalize for each batch
			# pred_val_idx.append(pred)			#for norm
			# feature_map_idx.append(xx)			#for norm
			
		# #print(np.shape(feature_map_idx))							#(20,1,28,28,8)
		# max_feature = np.amax(feature_map_idx, (0,1,2,3))
		# min_feature = np.amin(feature_map_idx, (0,1,2,3))			#minimum of each channel for 20 patches.
		# feature_range = max_feature - min_feature
		# # print(np.shape(max_feature), min_feature)
		# # quit()

		# pred_idx = 0
		# for feature_map in feature_map_idx:							#(1, 28, 28, 8)
		# 	pred_int = pred_val_idx[pred_idx].astype(int)
		# 	pred_idx += 1
		# 	norm_feature = feature_map.copy()

		# 	for e in range(8):
		# 		norm_feature[0,:,:,e] = (feature_map[0,:,:,e] - min_feature[e])/feature_range[e]
		# 		print(min_feature[e],feature_range[e])

		# 	save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, norm_feature, H_points, base_points)
		#################################

		###calculate mean err###
		err = np.mean(np.abs(pred_ori.astype(int).reshape(-1)-pred_int_dis))
		# if(sub_ave_mean != 0):
		# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
		# 	err_sub_ave_mean += sub_ave_mean
		# cnt_err += 1

		recorder.write('ERR: ' + str(err) + '  |||\r')
		
		# print('edge_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('edge_focus_mean_loss: {0:.2f}'.format(err_sub_ave_mean/cnt_err))
	recorder.close()



def val_bold(val_dir):

	for item in range(100):
		
		val_image, H_points, base_points, color_image, ori_img, deform_img = get_test(os.path.join(val_dir, 'pure.jpg'))
		save_dir = ''
		for e in H_points:
			save_dir += str(e)
			save_dir += '_'
		save_dir = save_dir[:-1]
		os.makedirs(save_dir)

		pred, boardline_feature = sess.run([logits, x1], feed_dict={datas: val_image.reshape([1,224,224,2])})
		reference_dir = os.path.join(save_dir, 'reference')
		os.makedirs(reference_dir)
		_ = save_fig(pred.astype(int).reshape(-1), reference_dir, ori_img, deform_img, ori_img, deform_img, boardline_feature, H_points, base_points, boardline_feature)

		########################
		print(' ========= {} ========= '.format(item))
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/edge/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
		################################
		# # Channel normalize for each batch
			# pred_val_idx.append(pred)			#for norm
			# feature_map_idx.append(xx)			#for norm
			
		# #print(np.shape(feature_map_idx))							#(20,1,28,28,8)
		# max_feature = np.amax(feature_map_idx, (0,1,2,3))
		# min_feature = np.amin(feature_map_idx, (0,1,2,3))			#minimum of each channel for 20 patches.
		# feature_range = max_feature - min_feature
		# # print(np.shape(max_feature), min_feature)
		# # quit()

		# pred_idx = 0
		# for feature_map in feature_map_idx:							#(1, 28, 28, 8)
		# 	pred_int = pred_val_idx[pred_idx].astype(int)
		# 	pred_idx += 1
		# 	norm_feature = feature_map.copy()

		# 	for e in range(8):
		# 		norm_feature[0,:,:,e] = (feature_map[0,:,:,e] - min_feature[e])/feature_range[e]
		# 		print(min_feature[e],feature_range[e])

		# 	save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, norm_feature, H_points, base_points)
		#################################

			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('edge_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('edge_focus_mean_loss: {0:.2f}'.format(err_sub_ave_mean/cnt_err))

		##########################################
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/dense/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('dense_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('dense_focus_mean_loss: {0:.2f}'.format(err_sub_ave_mean/cnt_err))

		##########################################
		err = 0
		err_sub_ave_mean = 0
		cnt_err = 0
		index = 0 

		loc_list = glob(val_dir+'/full/*')
		cnt_img = len(loc_list) -1

		for i in range(cnt_img):	
			val_images, gray_image, inv_warped_image, original_patch, warped_patch = get_test_visualization(loc_list, index, H_points, base_points)
			index += 1
			pred, xx = sess.run([logits, x1], feed_dict={datas: val_images.reshape([1,224,224,2])})
			#print(np.shape(xx))
			pred = np.around(np.array(pred), 2).reshape(-1)
			pred_int = pred.astype(int)
			
			sub_ave_mean = save_fig(pred_int, save_dir, gray_image, inv_warped_image, original_patch, warped_patch, xx, H_points, base_points, boardline_feature)
			
			###calculate mean err###
			err += np.mean(np.abs(pred-H_points))
			# if(sub_ave_mean != 0):
			# 	# print('focus_loss: {}, loss: {}'.format(sub_ave_mean, np.mean(np.abs(pred-H_points))))
			# 	err_sub_ave_mean += sub_ave_mean
			cnt_err += 1

		print('full_mean_loss: {0:.2f}'.format(err/cnt_err))
		# print('full_focus_mean_loss: {:.2f}'.format(err_sub_ave_mean/cnt_err))




epochs = 1000
batch_size = 64
data_batch = np.zeros((batch_size,256,256,1))
labels_batch = np.zeros((batch_size, 8))

#g = get_generator(path = "./eye/*.png", num_examples=batch_size, eye=0)

# logging.info( "Loading Val Data...")
# data_V, label_V = data_loader('./val.txt')
# data_V = np.asarray(data_V)
# label_V = np.asarray(label_V)


with tf.Graph().as_default():
	datas = tf.placeholder(tf.float32, (None, 224, 224, 2), name='data')
	labels = tf.placeholder(tf.float32, (None, 8), name='label')
	lr = tf.Variable(1e-4, name='learning_rate', trainable=False, dtype=tf.float32)
	
	# logits, x1 = homography_regression_resnet(datas)
	
	logits, x1 = homography_regression_model(datas)
	# logits, x1 = homography_regression_vgg(datas)

	loss = tf.reduce_mean(tf.square(tf.squeeze(logits) - tf.squeeze(labels)))
	
	tf.summary.scalar('loss',loss)

	opt = tf.train.AdamOptimizer(lr).minimize(loss)

	saver = tf.train.Saver(max_to_keep=None)
		
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())

		# saver.restore(sess, './H_predict/ckpt/ResNet/model299.ckpt')
		saver.restore(sess, './H_predict/ckpt/baseline/model499.ckpt')
		# saver.restore(sess, './H_predict/ckpt/VGG16/model349.ckpt')
		
		merged = tf.summary.merge_all()  
		writer = tf.summary.FileWriter(log_dir,sess.graph)  
		saver_all = tf.train.Saver(max_to_keep=1000)


		###testing###

		fig=plt.figure(figsize=(11,8)) 

		# Performance between edge | board | dense
		# val_bold("./val_edge_bold")

		# Performance between original real images & distortion images.
		# val_dis("isbi")

		# Simply get ERR, MAE
		val_err("hpatches")
		# val_err("BIWI")
		# val_err("MScoco_val2014")
		# for i in range(10):
		# 	val_err("GSS")

		


