import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as F
import torchvision.transforms as transforms

import torchmetrics


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, window_size):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()
        # Do you want to return ReLU insted of final vector?
        #self.ReLU_out = ReLU_out

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=500, kernel_size=5, padding=2)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=500, out_channels=100, kernel_size=2, padding=1)
        # Fully connected layer
        #self.fc1 = nn.Linear(100 * 7 * 11, num_classes)
        self.fc1 = nn.Linear(1, num_classes)  # temporary
        self._initialize_fc(in_channels, num_classes, window_size)

    def _initialize_fc(self, in_channels, num_classes, wind_size):
        # Build a dummy input to compute final flattened size
        x = torch.zeros(1, in_channels, wind_size, 20)   # adjust if your input size differs
        x = self.pool(self.conv1(x))
        x = F.relu(self.conv2(x))
        flat_size = x.numel()

        # Redefine fc1 with correct size
        self.fc1 = nn.Linear(flat_size, num_classes)

    def forward(self, x, ReLU_out=False):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = self.conv1(x)          # Apply first convolution without activation layer
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor into (batch_size,7700)

        if ReLU_out: # used only on trained CNN and in case we extract ReLU
            return x
        
        x = self.fc1(x)            # Apply fully connected layer
        return x
 
class Softmax(nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        pred = self.linear(x)
        return pred 