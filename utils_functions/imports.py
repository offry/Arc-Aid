import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import os.path
import shutil
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from shutil import move
from collections import OrderedDict
import matplotlib.pyplot as plt
import urllib
from PIL import Image, ImageOps
import random
import cv2
import wandb
from einops import reduce, rearrange
import requests
import pandas as pd
import seaborn as sns
from time import sleep
import argparse
import torchvision.transforms.functional as TF
import kornia
from utils_functions.util_functions import *
from util_models.util_models import *
from utils_functions.dataloaders_and_augmentations import *
from training.train import *
from training.train_shapes_with_periods import train_shapes_with_periods
from utils_functions.dataloaders_and_augmentations_periods_with_shapes import data_loader_both_shapes_with_periods
import torch.multiprocessing

Image.MAX_IMAGE_PIXELS = None