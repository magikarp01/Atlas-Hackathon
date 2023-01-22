# Want to separate the MNIST dataset by number
# Goal is to train learning net on part of dataset (e.g. only pictures of 1-6), 
# then have it additionally learn to classify pictures of 7

import torch 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
picture_dictionary = {}

# import the mnist training set
if __name__ == '__main__':
    for i in mnist_trainset:
        print(i[1])

class NumberDataset(Dataset):
    
