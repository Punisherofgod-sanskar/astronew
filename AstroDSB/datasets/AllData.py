# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB
# --------------------------------------------------------------------------------------------------

import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from scipy import ndimage
from sklearn.model_selection import train_test_split
# from astropy.io import fits

def build_train_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)  # [0,1] --> [-1, 1]
    ])

def build_test_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)  # [0,1] --> [-1, 1]
    ])

def build_astro_dataset(config, train=True):
    transforms = build_train_transform() if train else build_test_transform()

    data_all_array = np.load("PATH_TO_PREPROCESSED_ASTRO_DATA")

    X_data = data_all_array[0, :, :, :]  # input
    Y_data = data_all_array[1, :, :, :]  # target

    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.205, random_state=42)

    if train:
        X, Y = x_train, y_train
    else:
        X, Y = x_test, y_test

    dataset = AstroDataset(X=X, Y=Y, transforms=transforms)
    return dataset

class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transforms):
        super().__init__()
        self.X_data = X
        self.Y_data = Y
        self.transforms = transforms
        self.total_sample = X.shape[0]

    def __getitem__(self, index):
        input_img = self.X_data[index]
        target_img = self.Y_data[index]

        # Assuming input_img and target_img are numpy arrays [H, W, C]
        clean_img = self.transforms(target_img)  # clean (target)
        corrupt_img = self.transforms(input_img)  # corrupt (input)

        label = 0  # dummy label

        return clean_img, corrupt_img, label

    def __len__(self):
        return self.total_sample
