from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras import callbacks

#Setting parameter
batch_size = 50 #จำนวน data ที่ส่งไป Train ในแต่ละครั้ง 
width = 150 
height = 150 
input_shape = (height, width, 3) #ขนาด image enter
epochs = 2000  #จำนวนรอบในการ Train
NUM_TRAIN = 2914# จำนวนภาพ Train
NUM_TEST = 125 #จำนวนภาพ Test
dropout_rate = 0.2 















