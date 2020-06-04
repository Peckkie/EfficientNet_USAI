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
os.chdir('/media/tohn/SSD/FP_All_Nor_Abnor_B3NA_25/content/efficientnet_keras_transfer_learning/')
      #choose gpu on processing 
os.environ["CUDA_VISIBLE_DEVICES"]="0" # second gpu  

import sys
sys.path.append('/media/tohn/SSD/Nor_ABnor_Network_25/content/efficientnet_keras_transfer_learning')

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
model = load_model("/media/tohn/SSD/Nor_ABnor_Network_25/content/efficientnet_keras_transfer_learning/models/Nor_ABnor_b3_R2.h5")
model.summary()
model.pop()
model.summary()

##สร้าง Network ใหม่
model2 = models.Sequential()
model2.add(model)
model2.add(layers.Dense(1, activation='sigmoid', name="fc_out"))        #class --> 2
model2.summary()

##จัดการ data
import pandas as pd
import numpy as np
df1 = pd.read_csv (r'/home/yupaporn/EfficientNet_USAI/final_training_table.csv')
    #select data 
msk = np.random.rand(len(df1)) < 0.9
test = df1[~msk] 
train= df1[msk]
##การเเบ่งข้อมูล train/validation/test sets
    #The directory where we will
base_dir = './data/views'
os.makedirs(base_dir, exist_ok=True)
    # Directories for our training,validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
    # Directory with our training cat pictures
train_Nor_dir = os.path.join(train_dir, 'Normal')
os.makedirs(train_Nor_dir, exist_ok=True)
    # Directory with our training cat pictures
train_ABn_dir = os.path.join(train_dir, 'ABnormal')
os.makedirs(train_ABn_dir, exist_ok=True)
    # Directory with our training cat pictures
validation_Nor_dir = os.path.join(validation_dir, 'Normal')
os.makedirs(validation_Nor_dir, exist_ok=True)
    # Directory with our training cat pictures
validation_ABn_dir = os.path.join(validation_dir, 'ABnormal')
os.makedirs(validation_ABn_dir, exist_ok=True)
    
    #Path images of train
Nor_train = train[train['Class']=='Normal']
Nor_path_train = Nor_train['Path Crop'].tolist() 
ABn_train = train[train['Class']=='Abnormal']
ABn_path_train = ABn_train['Path Crop'].tolist() 

Nor_validation = test[test['Class']=='Normal']
Nor_path_validation = Nor_validation['Path Crop'].tolist() 
ABn_validation = test[test['Class']=='Abnormal']
ABn_path_validation= ABn_validation['Path Crop'].tolist() 

    #Train
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

print('Train images total : ',len(Nor_path_train)+len(ABn_path_train))
print('Validation images total : ',len(ABn_path_validation)+len(Nor_path_validation))
print('Total images : ',len(Nor_path_train)+len(ABn_path_train)+len(ABn_path_validation)+len(Nor_path_validation))

##Hyper parameters
batch_size = 50 

width = 150 
height = 150 
input_shape = (height, width, 3) #ขนาด image enter

epochs = 2000  #จำนวนรอบในการ Train
NUM_TRAIN = len(Nor_path_train)+len(ABn_path_train)  # จำนวนภาพ Train
NUM_TEST = len(ABn_path_validation)+len(Nor_path_validation) #จำนวนภาพ Test
dropout_rate = 0.2 #คือการปิดบาง Node หรือเรียกว่าทำการ Drop Out ไป ซึ่งขึ้นกับการตั้งค่าว่าจะให้ลืมไปกี่เปอร์เซนต์ดี ช่วยในการแก้ปัญหา Overfitting

##Setting data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255, 
      rotation_range=40, 
      width_shift_range=0.2, 
      height_shift_range=0.2, 
      shear_range=0.2, 
      zoom_range=0.2, 
      horizontal_flip=False, 
      fill_mode='nearest') 

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
root_logdir = '/media/tohn/SSD/FP_All_Nor_Abnor_B3NA_25/my_logs'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(run_logdir)

#ส้รางไฟล์เก็บโมเดล
os.makedirs("./models", exist_ok=True)

history = model2.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator, 
      validation_steps= NUM_TEST //batch_size,
      verbose=1, 
      use_multiprocessing=True, 
      workers=1,
      callbacks = [tensorboard_cb,callbacks.ModelCheckpoint(filepath='./models/FP_All_b325_call.h5', save_freq = 'epoch')])

hist_df = pd.DataFrame(history.history) 
hist_df.to_csv('hist_b3_All25.csv')

##save model    
model2.save('./models/FP_All_b325.h5')

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
plt.savefig('plot_acc_FP_All25_.png')

plt.figure()
plt.plot(epochs_x, loss, 'ro', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
    #save plot_loss
plt.savefig('plot_loss_FP_All25_.png')

