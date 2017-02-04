###############################################################################
# Author: Abhimanyu Banerjee
# Project: Deep Image Colorizer
# Date Created: 2/3/2017
# 
# File Description: This script builds a simple neural network consisting of 
# several layers of downsampling followed by several layers of upsampling to 
# learn to colorize grayscale images provided as input to the model.
###############################################################################

import os
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pickle
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ColorizerNet(nn.Module):

	def __init__(self):
		super(ColorizerNet, self).__init__()
		self.layer1 = nn.Conv2d(1, 8, 2, 2)
		self.layer2 = nn.Conv2d(8, 16, 2, 2)
		self.layer3 = nn.Conv2d(16, 8, 2, 2)
		self.layer4 = nn.Conv2d(8, 1, 2, 2)


	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))
		return x

if __name__ == '__main__':
	
	colorizerNet = ColorizerNet()
	print(colorizerNet)