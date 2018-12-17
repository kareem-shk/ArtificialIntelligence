## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU() 
        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.ReLU() 
        self.pooling2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.ReLU() 
        self.pooling3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.activation4 = nn.ReLU() 
        self.pooling4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.activation5 = nn.ReLU() 
        self.pooling5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.activation6 = nn.ReLU() 
        self.pooling6 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=0)
        self.activation7 = nn.ReLU() 

        self.fc1 = nn.Linear(2048, 136)
        #self.activation5 = nn.ReLU()
        
        #self.fc2 = nn.Linear(1024, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.pooling1(y)

        y = self.conv2(y)
        y = self.activation2(y)
        y = self.pooling2(y)
        
        y = self.conv3(y)
        y = self.activation3(y)
        y = self.pooling3(y)

        y = self.conv4(y)
        y = self.activation4(y)
        y = self.pooling4(y)

        y = self.conv5(y)
        y = self.activation5(y)
        y = self.pooling5(y)
        
        y = self.conv6(y)
        y = self.activation6(y)
        y = self.pooling6(y)

        y = self.conv7(y)
        y = self.activation7(y)
        
        y = y.view(y.size(0), -1)
        y = self.fc1(y)   
        #y = self.activation5(y)

        #y = self.fc2(y)           
        return y
