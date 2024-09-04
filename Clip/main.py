import pathlib
import torch
import torch.nn as nn

import torchvision.transforms as transforms

import datasets
import argparse

'''default setting of the model'''
parser = argparse.ArgumentParser(description='Clip Network')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available else 'cpu'

'''hyperparameters setting'''
epoch = 0
max_epoch = 1000

modalities = ["T1", "T2"]
