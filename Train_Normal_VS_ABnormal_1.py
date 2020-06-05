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

import os
    # os.path.abspath('/media/tohn/SSD/Efficient_USAI')
os.chdir('/media/tohn/SSD/test_func/content/efficientnet_keras_transfer_learning/')
      #choose gpu on processing 
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu  

import sys
sys.path.append('/media/tohn/SSD/Sub_Efficient_USAI/content/efficientnet_keras_transfer_learning')

##load model 
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

from tensorflow.keras.models import load_model
model = load_model("/media/tohn/SSD/Sub_Efficient_USAI/content/efficientnet_keras_transfer_learning/models/Sub_b0_c11_250_R2.h5")
model.summary()
model.pop()
model.summary()

##สร้าง Network ใหม่
# model2 = models.Sequential()
# model2.add(model)
# model2.add(layers.Dense(1, activation='sigmoid', name="fc_out"))        #class --> 2
# model2.summary()
x = model.output
prediction_layer = layers.Dense(1, activation='sigmoid')(x)
model2 = models.Model(inputs=model.inputs, outputs=prediction_layer)

##จัดการ data
import pandas as pd
df = pd.read_csv (r'/home/yupaporn/EfficientNet_USAI/final_training_table.csv')

msk = np.random.rand(len(df)) < 0.9 #split
test = df[~msk]
train= df[msk]

##การเเบ่งข้อมูล train/validation/test sets

    #The directory where we will
base_dir = './data/views'
os.makedirs(base_dir, exist_ok=True)

    # Directories for our training,
    # validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)

    # Directory with our training pictures
train_Nor_dir = os.path.join(train_dir, 'Normal')
os.makedirs(train_Nor_dir, exist_ok=True)
train_ABn_dir = os.path.join(train_dir, 'ABnormal')
os.makedirs(train_ABn_dir, exist_ok=True)
    # Directory with our validation pictures
validation_Nor_dir = os.path.join(validation_dir, 'Normal')
os.makedirs(validation_Nor_dir, exist_ok=True)
validation_ABn_dir = os.path.join(validation_dir, 'ABnormal')
os.makedirs(validation_ABn_dir, exist_ok=True)

    #Path images of train
Nor_train = train[train['Class']=='Normal']
Nor_path_train = Nor_train['Path Crop'].tolist() 
ABn_train = train[train['Class']=='Abnormal']
ABn_path_train = ABn_train['Path Crop'].tolist() 
    #Path images of validation
Nor_validation = test[test['Class']=='Normal']
Nor_path_validation = Nor_validation['Path Crop'].tolist() 
ABn_validation = test[test['Class']=='Abnormal']
ABn_path_validation= ABn_validation['Path Crop'].tolist() 

    #Training
fnames = Nor_path_train  
for fname in fnames:
    dst = os.path.join(train_Nor_dir, os.path.basename(fname))
    if os.path.exists(dst):
        dst = dst+'cp.jpg'
    shutil.copyfile(fname, dst)
    
fnames = ABn_path_train  
for fname in fnames:
    dst = os.path.join(train_ABn_dir, os.path.basename(fname))
    if os.path.exists(dst):
        dst = dst+'cp.jpg'
    shutil.copyfile(fname, dst)
  
     #Validation
fnames = Nor_path_validation 
for fname in fnames:
    dst = os.path.join(validation_Nor_dir, os.path.basename(fname))
    if os.path.exists(dst):
        dst = dst+'cp.jpg'
    shutil.copyfile(fname, dst)

fnames = ABn_path_validation
for fname in fnames:
    dst = os.path.join(validation_ABn_dir, os.path.basename(fname))
    if os.path.exists(dst):
        dst = dst+'cp.jpg'
    shutil.copyfile(fname, dst)

##Hyper parameters
batch_size = 50 

width = 150 
height = 150 
input_shape = (height, width, 3) #ขนาด image enter

epochs = 2000
NUM_TRAIN = len(Nor_path_train)+len(ABn_path_train)  
NUM_TEST = len(ABn_path_validation)+len(Nor_path_validation) 
dropout_rate = 0.2

##Setting data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255, # image input 0-255 --> 0-1 เปลี่ยนค่าสี
      rotation_range=40, # หมุนภาพในองศา
      width_shift_range=0.2, #เปลี่ยนความกว้าง
      height_shift_range=0.2, #ปลี่ยนความสูง
      shear_range=0.2, #ทำให้ภาพเบี้ยว
      zoom_range=0.2, #ซุม image มากสุด 20%
      horizontal_flip=False, #พลิกภาพแบบสุ่มตามแนวนอน
      fill_mode='nearest') 

    # Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='binary')

##Training
model2.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
    
    #สร้าง folder TensorBoard
root_logdir = '/media/tohn/SSD/Nor_ABnor_Network/my_logs'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(run_logdir)

history = model2.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator, 
      validation_steps= NUM_TEST //batch_size,
      verbose=1, 
      use_multiprocessing=True, 
      workers=1,
      callbacks = [tensorboard_cb,callbacks.EarlyStopping(monitor='val_acc', patience=50, mode='max'), callbacks.ModelCheckpoint(filepath='./models/ABn_vs_Nor_call.h5' , save_freq = 'epoch')])

##save model    
os.makedirs("./models", exist_ok=True)
model2.save('./models/ABn_vs_Nor_func.h5')

##plot graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'ro', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
    #save plot_acc
plt.savefig('plot_acc_ABn_vs_Nor_func.png')

plt.figure()
plt.plot(epochs_x, loss, 'ro', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
    #save plot_loss
plt.savefig('plot_loss_ABn_vs_Nor_func.png')
















