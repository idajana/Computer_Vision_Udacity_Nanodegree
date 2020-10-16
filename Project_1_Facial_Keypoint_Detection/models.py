## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, use_dropout=False):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        if use_dropout:
            prob = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            prob = [0.0, 0.0, 0.0, 0.0, 0.0]        
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Output conv1 (32, 220, 220)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Output pool1 (32, 110, 110)
        self.dropout1 = nn.Dropout(p=prob[0])
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Output conv2 (64, 108, 108)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Output pool2 (64, 54, 54)
        self.dropout2 = nn.Dropout(p=prob[1])
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Output conv3 (128, 52, 52)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Output pool3 (128, 26, 26)        
        self.dropout3 = nn.Dropout(p=prob[2])
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        # Output conv4 (256, 24, 24)
        self.pool4 = nn.MaxPool2d(2, 2)
        # Output pool4 (256, 12, 12)        
        self.dropout4 = nn.Dropout(p=prob[3])
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        # Output conv5 (512, 10, 10)
        self.pool5 = nn.MaxPool2d(2, 2)
        # Output pool5 (512, 5, 5)        
        self.dropout5 = nn.Dropout(p=prob[4])
        
        self.fc6 = nn.Linear(512*5*5, 5000)
        self.dropout6 = nn.Dropout(p=prob[2])
        
        self.fc7 = nn.Linear(5000, 1500)
        self.dropout7 = nn.Dropout(p=prob[3])
        
        self.fc8 = nn.Linear(1500, 136)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.elu(self.conv4(x))))
        x = self.dropout5(self.pool5(F.elu(self.conv5(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout6(self.fc6(x))
        x = self.dropout7(self.fc7(x))
        x = self.fc8(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
