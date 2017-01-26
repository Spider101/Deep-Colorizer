###############################################################################
# Author: Abhimanyu Banerjee
# Project: Deep Image Colorizer
# Date Created: 1/22/2017
# 
# File Description: This script converts the data downloaded from the Labelled 
# Faces in the Wild ('lfw') website into a robust dataset that can be used for
# training a image-colorizer model based on deep neural nets.
###############################################################################

import os
from os import listdir
from os.path import isfile, join, isdir, isfile
import shutil
import urllib.request
import tarfile
from skimage.io import imread
from skimage.transform import rotate, resize
from skimage.color import rgb2lab
import numpy as np
import pickle
import pdb

'''the images of interested are nested within two levels in the 'lfw' directory.
To create a consolidated dataset, the images removed from the subdirectories 
and placed in a 'data' directory. Finally, the original 'lfw directory is 
removed.'''
def extractImages(faces_dir, data_dir):
	#initialize the relevant path variables
	faces_dir_path = join(os.getcwd(), faces_dir)
	data_dir_path = join(os.getcwd(), data_dir)
	sub_dirs = [sub_dir for sub_dir in listdir(faces_dir_path)]

	#remove the data directory if it exists to maintain consistency
	if isdir(data_dir_path):
		shutil.rmtree(data_dir_path)
	
	os.mkdir(data_dir_path)
	
	#cycle through the sub-directories in the Labelled Faces in the Wild
	#directory and extract the images out to the 'data' directory
	print("\nPopulating the Data directory...")
	for sub_dir in sub_dirs:
		sub_dir_path = join(faces_dir_path, sub_dir)
		images = listdir(sub_dir_path)
		for image in images:
		    shutil.copy(join(sub_dir_path, image), data_dir_path)

	#dispose of the original dataset to avoid redundancy
	shutil.rmtree(faces_dir_path)
	print("\nData directory has been setup. Original directory removed!")

'''load the images in the 'data' directory into numpy arrays for ease of 
computation and manipulation'''
def loadDataset(data_dir_path):
	print("\nCommencing Loading of Data..")
	images = [image for image in listdir(data_dir_path)]
	dataset = []
	for image in images:
		image = imread(join(data_dir_path, image))
		resized_image = resize(image, (128, 128)) # resize images to 128x128
		dataset.append(resized_image)
	print("Dataset Loaded!")
	return np.array(dataset) 

'''augments the faces dataset by applying flips - both vertical and horizontal, 
rotations and random crops. Since the task at hand is estimating the color of 
grayscale images, manipulation of contrast or saturation of the color channels
is avoided.'''
def augmentDataset(faces_data, pickle_path):
	num_images, width, height, num_channels = faces_data.shape[0], faces_data.shape[1], faces_data.shape[2], faces_data.shape[3]
	#pdb.set_trace()
	aug_dataset = np.zeros((6*num_images, width, height, num_channels))

	print("\nCommencing augmentation of data")
	for i in range(num_images):
		aug_dataset[6*i, :, :, :] = faces_data[i, :, :, :]
		aug_dataset[6*i + 1, :, :, :] = np.fliplr(faces_data[i, :, :, :]) #flip horizontally
		aug_dataset[6*i + 2, :, :, :] = np.flipud(faces_data[i, :, :, :]) #flip vertically
		aug_dataset[6*i + 3, :, :, :] = rotate(faces_data[i, :, :, :], 90)
		aug_dataset[6*i + 4, :, :, :] = rotate(faces_data[i, :, :, :], 180)
		aug_dataset[6*i + 5, :, :, :] = rotate(faces_data[i, :, :, :], 270)
		#aug_dataset[6*i + 7, :, :, :] = faces_data[i, :, :, :]
	print("Data Augmentation Complete!")

	#storing the augmented data to avoid repeating this process over and over again
	#pickle.dump(aug_dataset, open(pickle_path, "wb")) 
	np.savez("augData", data=aug_dataset)
	print("\nData store created!")
	return aug_dataset

'''prepares the data to be in a form that can used for training the DNN model -
converts the colorspace to LAB, sets the first channel as the training features
and the mean of the other two as the training labels and normalizes the values'''
def prepareData(faces_data):
	num_images = faces_data.shape[0]
	lab_data = np.zeros_like(faces_data)

	#converting to LAB colorspace
	print("\nConverting to LAB colorspace. Please wait...")
	for i in range(num_images):
		lab_data[i, :, :, :] = rgb2lab(faces_data[i, :, :, :])
	print("Colorspace conversion complete!")

	#separate the data into features and labels
	features = lab_data[:, :, :, 0:1]
	labels = lab_data[:, :, :, 1:]
	pdb.set_trace()

	#normalizing the features
	for i in range(num_images):
		feature_max = np.amax(features[i])
		feature_min = np.amin(features[i])
		feature_range = feature_max - feature_min

		features[i] = (features[i] - feature_min) / feature_range

	return {"features" : features, "labels": labels} 

if __name__ == "__main__":
	faces_dir = "lfw"
	data_dir = "data"

	#check if the data directory is setup
	if not isdir(join(os.getcwd(), data_dir)):
		
		#check if Labelled Faces in the Wild ('lfw') directory exists and if not 
		#downloads the .tgz file from the associated website, unzips it and extracts
		#the images to a 'data' directory to form a consolidated dataset
		if not isdir(join(os.getcwd(), faces_dir)):
			url, file_name = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz', "lfw.tgz" 
			
			#download the dataset
			print("\nDownloading the dataset. Please wait...")
			urllib.request.urlretrieve(url, file_name)
			print("Download Complete!")
			
			print("\nExtracting the dataset. Please wait...")
			t = tarfile.open('lfw.tgz', 'r')
			t.extractall()
			print("Dataset Extracted!")

		#move the images from the lfw directory to the data directory
		extractImages(faces_dir, data_dir)

	#check if pickle for augmented data exists
	pickle_path = join(os.getcwd(), "augData.npz")
	if isfile(pickle_path):
		print("\nLoading data store..")
		dataset = np.load("augData.npz")["data"]
		print("Data store loaded!")

	#no pickle exists, so proceed to make the pickle
	else:
		print("\nNo previously stored data found. Preparing data store...")
	
		#load the data into numpy arrays
		data_dir_path = join(os.getcwd(), data_dir)
		dataset = loadDataset(data_dir_path)

		#augment the dataset by a factor of 6
		dataset = augmentDataset(dataset, pickle_path)

	final_dataset = prepareData(dataset)


