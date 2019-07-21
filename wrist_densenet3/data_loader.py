import os
import numpy as np
import cv2
import random
import keras.backend as K

def load_path(root_path = '../valid/XR_ELBOW', size = 512):
	'''
	load MURA data


	'''

	Path = []
	labels = []
	for root,dirs,files in os.walk(root_path):
		for name in files:
			path_1 = os.path.join(root,name)
			Path.append(path_1)
			if 'yes' in path_1 or 'positive' in path_1:
				labels+=[1]   	          	 #patient11880\\study1_negative\\image3.png
			else:
			    labels+=[0]
	print (len(Path))
	labels = np.asarray(labels)
	return Path, labels

def load_image(Path = '../valid/XR_ELBOW', size = 512):
	Images = []
	for path in Path:
		if path.endswith('.jpg'):
			image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image,(size,size))
			image = randome_rotation_flip(image,size)
			Images.append(image)

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	Images = np.repeat(Images[..., np.newaxis], 3, -1)

	# if K.image_data_format() == "channels_first":
	# 	Images = np.expand_dims(Images,axis=1)
	# if K.image_data_format() == "channels_last":
	# 	Images = np.expand_dims(Images,axis=3)             #(usebackend tensorflow:aixs=3; theano:axixs=1)

	return Images


def randome_rotation_flip(image,size = 512):
	if random.randint(0,1):
		iamge = cv2.flip(image,1)

	if random.randint(0,1):
		angle = random.randint(-30,30)
		M = cv2.getRotationMatrix2D((size/2,size/2),angle,1)
		image = cv2.warpAffine(image,M,(size,size))
	return image



if __name__ == '__main__':
	load_path()
