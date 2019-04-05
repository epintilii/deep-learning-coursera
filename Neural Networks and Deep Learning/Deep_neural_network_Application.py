import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
