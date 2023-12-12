# ALMS 1 ELEC0134
# Term 1 Final Assignment
# By SN 23043422, from 13/12/2023 to ??/01/2024

# This file aim to address all main code parts used when solving task A and B described in the task sheet;

# Environment: Python 3.12.0

# Library installed (as the order to be used): medmnist

# Datasets for both tasks, for task A and for task B respectively, are installed by
# %pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

# Task A

#importing libraries
import medmnist
import numpy as np
#print("successfully installed madmnist, version:", medmnist.__version__)

#importing training, validating and testing data seperately
data = np.load('Dataset/pneumoniamnist.npz')
data_train = data['train_images']
print(len(data_train))
# from medmnist import PneumoniaMNIST
# dataset_train1 = PneumoniaMNIST(split="train", download=True)
# dataset_val1 = PneumoniaMNIST(split="val", download=True)
# dataset_test1 = PneumoniaMNIST(split="test", download=True)
# print(dataset_val1)