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
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

class ColorizerNet(nn.Module):

	def __init__(self):
		super(ColorizerNet, self).__init__()
		self.layer1 = nn.Conv2d(1, 8, 2, 2)
		self.layer2 = nn.Conv2d(8, 16, 2, 2)
		self.layer3 = nn.ConvTranspose2d(16, 8, 2, 2)
		self.layer4 = nn.ConvTranspose2d(8, 2, 2, 2)


	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))
		return x

def trainNet(model, data_loader, criterion, num_epochs):

	print("\nStarting to train model. Please wait..")
	for epoch in range(num_epochs): # loop over the dataset multiple times
    
	    running_loss = 0.0
	    for i, data in enumerate(data_loader, 0):
	        # get the inputs
	        inputs, labels = data
	        
	        #convert the data to floats for ease of use on GPU
	        inputs = inputs.float()
	        labels = labels.float()
	        
	        # wrap them in Variable
	        inputs, labels = Variable(inputs), Variable(labels)
	        
	        # zero the parameter gradients
	        optimizer.zero_grad()
	        
	        # forward + backward + optimize
	        
	        outputs = model(inputs)
	        #pdb.set_trace()
	        loss = criterion(outputs, labels)
	        loss.backward()        
	        optimizer.step()
	        
	        # print statistics
	        running_loss += loss.data[0]
	        if i % 2000 == 0: # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
	            running_loss = 0.0

	print('Finished Training')

if __name__ == '__main__':
	
	#relevant path names
	feature_path = join(os.getcwd(), "data", "features.npz")
	label_path = join(os.getcwd(), "data", "labels.npz")

	#load the features and labels as tensors with appropriate shape for torch
	print("\nLoading the features set.Please wait ..")
	features = np.load(feature_path)["features"]
	features = np.rollaxis(features, 3, 1)
	features = torch.from_numpy(features)

	print("\nLoading the  corresponding labels. Please wait ..")
	labels = np.load(label_path)["labels"]
	labels = np.rollaxis(labels, 3, 1)
	labels = torch.from_numpy(labels)
	
	#pdb.set_trace()

	#split data into train and test set and create data loaders for each
	trainset_prop = int(0.7*features.size()[0])
	trainset = TensorDataset(features[:trainset_prop], labels[:trainset_prop])
	testset = TensorDataset(features[trainset_prop:], labels[trainset_prop:])

	trainset_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
	testset_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)	

	#setup model, optimizer and criterion
	colorizerNet = ColorizerNet()
	criterion = nn.MSELoss() # use a Classification Cross-Entropy loss
	optimizer = optim.SGD(colorizerNet.parameters(), lr=0.001, momentum=0.9)
	print(colorizerNet)

	trainNet(colorizerNet, trainset_loader, criterion, 5)
	
	
	