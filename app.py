#!/usr/bin/env python3

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.utils as utils

import numpy
import sys

#set all transforms on a dataset
transform_train = transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.RandomRotation( 10 ),
                                   transforms.Resize( 150 ),
                                   transforms.CenterCrop( 128 ),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])
])


dataset_train =    datasets.ImageFolder( "data/train", transform = transform_train )
dataloader_train = utils.data.DataLoader( dataset_train, batch_size = 32, shuffle = True )

model = nn.Sequential(
        
        # first input layer
        nn.Conv2d( 1, 8, 5 ),    # weights can't be random
        nn.ReLU(),                 
        nn.MaxPool2d( 2, 2 ),

        # second input layer
        nn.Conv2d( 8, 32, 5 ),  # weughts in second layer must be the same as in the first input layer
        nn.ReLU(),              # similar activation function in both input layers
        nn.MaxPool2d( 2, 2 ),
        
        nn.Flatten( start_dim = 1 ),
        
        #hidden layers
        nn.Linear( 32 * 29 * 29, 1024 ), # weights w1, biases b1 
        nn.ReLU(),

        nn.Linear( 1024, 256 ),          # weights w2, biases b2
        nn.ReLU(),
        # V
        # w2 = f(w1), b2 = f(b1)



        #output layer
        nn.Linear( 256, 2 ),             # weigts w3, biases b3

        )                                #

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam( model.parameters(), lr = 0.001, weight_decay = 0.0001 )

print( model )










