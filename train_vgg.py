import tensorflow as tf 
import numpy as np 
import os, os.path
import cv2, re, random
# import oct, deformation
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from glob import glob
from numpy.linalg import inv

from scipy import stats
from multiprocessing import Process, Queue
import datetime, time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

run_id = 'VGG16'
log_dir = './H_predict/log/{}'.format(run_id)
ckpt_dir = './H_predict/ckpt/{}'.format(run_id)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
import logging

logging.basicConfig(filename=log_dir + '/H_predict.log',
														filemode='a', level=logging.DEBUG,
														format='%(asctime)s, %(msecs)d %(message)s',
														datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#############################
logging.info('\n\nH_predict...\n') 

######## Pre-method ############

def homography_regression_model(input_img):
	#input shape(N, 224, 224, 2)
	
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
	out = tf.reduce_mean(x, axis=[1,2])
	'''

	x = tf.layers.flatten(x)
	x = tf.layers.dropout(x, rate=0.75)
	x = tf.layers.dense(x, 1024)
	out = tf.layers.dense(x, 8)
	'''
	return out


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
		
	return out


#############################################################



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

	return out
#############################################################

def pre_deform(path, num_examples = 256):
	loc_list = glob(path)
	X = []
	Y = []
	for i in range(num_examples):
		index = random.randint(0, len(loc_list)-1)
		img_file_location = loc_list[index]
		color_image = cv2.imread(img_file_location)
			
		try:
			gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
		except Exception as e:
			print(img_file_location)
			continue
		ori, deform = deformation.Deform(gray_image, 10, img_size=512)
		X.append(ori)
		Y.append(deform)
	return X, Y

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
	loc_list = glob(path)
	index = random.randint(0, len(loc_list)-1)
	img_file_location = loc_list[index]


	# color_image = cv2.imread("/home/shao/workspace/.jpg")

	# #For *.png
	# if(img_file_location.split('.')[-1] == 'png'):
	# 	color_image = np.array(cv2.imread(img_file_location))		# Why use np.array(Image.open(img_file_location)) for *.png ?
	# else:
	color_image = cv2.imread(img_file_location)

	try:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
	except Exception as e:
		print(img_file_location)
	
	gray_image = cv2.resize(gray_image, (width, height))
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


def consumer(queue):
	X, Y = queue.get(block = True)
	return X, Y


epochs = 500
batch_size = 128
data_batch = np.zeros((batch_size,256,256,1))
labels_batch = np.zeros((batch_size, 8))

logging.info("Loading Train Data...")

#g = get_generator(path = "./eye/*.png", num_examples=batch_size, eye=0)

# logging.info( "Loading Val Data...")
# data_V, label_V = data_loader('./val.txt')
# data_V = np.asarray(data_V)
# label_V = np.asarray(label_V)

queue = Queue(maxsize = 300)
processes = [Process(target = get_generator, args = (queue, "./train/*.jpg", batch_size)) for x in range(8)]

for p in processes:
	p.start()


with tf.Graph().as_default():
	datas = tf.compat.v1.placeholder(tf.float32, (None, 224, 224, 2), name='data')
	labels = tf.compat.v1.placeholder(tf.float32, (None, 8), name='label')
	lr = tf.Variable(1e-4, name='learning_rate', trainable=False, dtype=tf.float32)
	
	logits = homography_regression_vgg(datas)

	loss = tf.reduce_mean(tf.square(tf.squeeze(logits) - tf.squeeze(labels)))
	
	tf.compat.v1.summary.scalar('loss',loss)

	opt = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

	saver = tf.compat.v1.train.Saver(max_to_keep=None)
	
	#logging.info("training start_now")
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.local_variables_initializer())
		sess.run(tf.compat.v1.global_variables_initializer())

		# saver.restore(sess, './H_predict/ckpt/baseline/model349.ckpt')
		
		merged = tf.compat.v1.summary.merge_all()  
		writer = tf.compat.v1.summary.FileWriter(log_dir,sess.graph)  
		saver_all = tf.compat.v1.train.Saver(max_to_keep=1000)

		for e in range(epochs):
	#		batch_x, batch_y = consumer(queue) #next(g)
			
			for item in range(100):
				batch_x, batch_y = consumer(queue) #next(g)
				if(item % 10 == 0):
					print(item,end=', ')
				summary, _, cl = sess.run([merged, opt, loss], feed_dict={datas: batch_x, labels: batch_y})
				#print(str(item)+',',end='')		#Doesn't work?
				#print(item)
				
			logging.info(' ========= {} ========= '.format(e))
			logging.info('training_loss: {}'.format(cl))
			
			if((e+1)%50 == 0):
				save_path = saver_all.save(sess, "{}/model{}.ckpt".format(ckpt_dir, e))
				print ('Model saved in file: %s' % save_path)
				
				err = 0
				for item in range(100):
					val_image, H_points, base_points, color_image, ori_img, deform_img = get_test("./test/*.jpg")
					pred = sess.run([logits], feed_dict={datas: val_image.reshape([1,224,224,2])})
					pred = np.array(pred)#.reshape((4,2))
					err += np.mean(np.abs(pred-H_points))
				logging.info('testing_loss: {}'.format(err/100))
	
		queue.close()
		for p in processes:
			p.terminate()
		quit()
