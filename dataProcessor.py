###############################################################################
# Author: Abhimanyu Banerjee
# Project: Deep Image Colorizer
# Date Created: 1/22/2017
# 
# File Description: This script checks if Labelled Faces in the Wild ('lfw') 
# directory exists and if not downloads the .tgz file from the associated 
# website and unzips it. It then cycles through the sub-directories of 'lfw' 
# directory to extract all the images of the faces out to the 'data' directory 
# to form a consolidated data-set and then removes the 'lfw' directory.
###############################################################################
import os
from os import listdir
from os.path import isfile, join, isdir
import shutil
import urllib.request
import tarfile

faces_dir = join(os.getcwd(), "lfw")
sub_dir_path = join(os.getcwd(), faces_dir)
data_dir_path = join(os.getcwd(), "data")

#check if the Labelled Faces in the Wild directory exists
if not isdir(faces_dir):
	url, file_name = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz', "lfw.tgz" 
	urllib.request.urlretrieve(url, file_name)
	t = tarfile.open('lfw.tgz', 'r')
	t.extractall()

sub_dirs = [sub_dir for sub_dir in listdir(faces_dir)]

#cycle through the sub-directories in the Labelled Faces in the Wild
#directory and extract the images out to the 'data' directory
for sub_dir in sub_dirs:
    images = listdir(join(sub_dir_path, sub_dir))
    for image in images:
        shutil.copy(join(sub_dir_path, sub_dir, image), data_dir_path)

#check if data transfer was successful
num_faces = len(listdir(data_dir_path))
print(num_faces)

#dispose of the original dataset
shutil.rmtree(join(os.getcwd(), faces_dir))


