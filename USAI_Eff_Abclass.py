import pandas as pd
import shutil

df = pd.read_csv (r'/home/yupaporn/codes/USAI/detail1_250.csv')

import os
# os.path.abspath('/media/tohn/SSD/Efficient_USAI')
os.chdir('/media/tohn/SSD/Efficient_USAI/content/efficientnet_keras_transfer_learning/')

# The directory where we will
base_dir = './data/views'
os.makedirs(base_dir, exist_ok=True)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Directory with our training cat pictures
validation_P1_dir = os.path.join(validation_dir, 'P1')
os.makedirs(validation_P1_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P2_dir = os.path.join(validation_dir, 'P2')
os.makedirs(validation_P2_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P3_dir = os.path.join(validation_dir, 'P3')
os.makedirs(validation_P3_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P4_dir = os.path.join(validation_dir, 'P4')
os.makedirs(validation_P4_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P5_dir = os.path.join(validation_dir, 'P5')
os.makedirs(validation_P5_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P6_dir = os.path.join(validation_dir, 'P6')
os.makedirs(validation_P6_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P7_dir = os.path.join(validation_dir, 'P7')
os.makedirs(validation_P7_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P8_dir = os.path.join(validation_dir, 'P8')
os.makedirs(validation_P8_dir, exist_ok=True)

# Directory with our training cat pictures
train_P1_dir = os.path.join(train_dir, 'P1')
os.makedirs(train_P1_dir, exist_ok=True)
# Directory with our training cat pictures
train_P2_dir = os.path.join(train_dir, 'P2')
os.makedirs(train_P2_dir, exist_ok=True)
# Directory with our training cat pictures
train_P3_dir = os.path.join(train_dir, 'P3')
os.makedirs(train_P3_dir, exist_ok=True)
# Directory with our training cat pictures
train_P4_dir = os.path.join(train_dir, 'P4')
os.makedirs(train_P4_dir, exist_ok=True)
# Directory with our training cat pictures
train_P5_dir = os.path.join(train_dir, 'P5')
os.makedirs(train_P5_dir, exist_ok=True)
# Directory with our training cat pictures
train_P6_dir = os.path.join(train_dir, 'P6')
os.makedirs(train_P6_dir, exist_ok=True)
# Directory with our training cat pictures
train_P7_dir = os.path.join(train_dir, 'P7')
os.makedirs(train_P7_dir, exist_ok=True)
# Directory with our training cat pictures
train_P8_dir = os.path.join(train_dir, 'P8')
os.makedirs(train_P8_dir, exist_ok=True)

df1 =df[df['Path Crop'] != 'None']

#Create Path images
val = df1[df1['Case'].between(1, 10)]
train = df1[df1['Case'].between(11, 250)]

#Path images of validation
p1_val = val[val['Abs Position']=='P1' ]
p1_path_val = p1_val['Path Crop'].tolist() 
p2_val = val[val['Abs Position']=='P2' ]
p2_path_val = p2_val['Path Crop'].tolist() 
p3_val = val[val['Abs Position']=='P3' ]
p3_path_val = p3_val['Path Crop'].tolist() 
p4_val = val[val['Abs Position']=='P4' ]
p4_path_val = p4_val['Path Crop'].tolist() 
p5_val = val[val['Abs Position']=='P5' ]
p5_path_val = p5_val['Path Crop'].tolist() 
p6_val = val[val['Abs Position']=='P6' ]
p6_path_val = p6_val['Path Crop'].tolist() 
p7_val = val[val['Abs Position']=='P7' ]
p7_path_val = p7_val['Path Crop'].tolist() 
p8_val = val[val['Abs Position']=='P8' ]
p8_path_val = p8_val['Path Crop'].tolist() 

#Path images of train
p1_train = train[train['Abs Position']=='P1' ]
p1_path_train = p1_train['Path Crop'].tolist() 
p2_train = train[train['Abs Position']=='P2' ]
p2_path_train = p2_train['Path Crop'].tolist() 
p3_train = train[train['Abs Position']=='P3' ]
p3_path_train = p3_train['Path Crop'].tolist() 
p4_train = train[train['Abs Position']=='P4' ]
p4_path_train = p4_train['Path Crop'].tolist() 
p5_train = train[train['Abs Position']=='P5' ]
p5_path_train = p5_train['Path Crop'].tolist() 
p6_train = train[train['Abs Position']=='P6' ]
p6_path_train = p6_train['Path Crop'].tolist() 
p7_train = train[train['Abs Position']=='P7' ]
p7_path_train = p7_train['Path Crop'].tolist() 
p8_train = train[train['Abs Position']=='P8' ]
p8_path_train = p8_train['Path Crop'].tolist() 

# copy images each view to validation_view_dir

fnames = p1_path_val  
for fname in fnames:
    dst = os.path.join(validation_P1_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p2_path_val  
for fname in fnames:
    dst = os.path.join(validation_P2_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p3_path_val  
for fname in fnames:
    dst = os.path.join(validation_P3_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p4_path_val  
for fname in fnames:
    dst = os.path.join(validation_P4_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)  
    
fnames = p5_path_val  
for fname in fnames:
    dst = os.path.join(validation_P5_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p6_path_val  
for fname in fnames:
    dst = os.path.join(validation_P6_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p7_path_val  
for fname in fnames:
    dst = os.path.join(validation_P7_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p8_path_val  
for fname in fnames:
    dst = os.path.join(validation_P8_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

# copy images each view to train_view_dir

    fnames = p1_path_train  
for fname in fnames:
    dst = os.path.join(train_P1_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p2_path_train  
for fname in fnames:
    dst = os.path.join(train_P2_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p3_path_train  
for fname in fnames:
    dst = os.path.join(train_P3_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p4_path_train  
for fname in fnames:
    dst = os.path.join(train_P4_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p5_path_train  
for fname in fnames:
    dst = os.path.join(train_P5_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p6_path_train  
for fname in fnames:
    dst = os.path.join(train_P6_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p7_path_train  
for fname in fnames:
    dst = os.path.join(train_P7_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p8_path_train  
for fname in fnames:
    dst = os.path.join(train_P8_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu  

    
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image

#only have 2 label
batch_size = 50 #จำนวน data ที่ส่งไป Train ในแต่ละครั้ง จนครบจำนวนเต็ม x_train

width = 150 
height = 150 
input_shape = (height, width, 3) #ขนาด image enter

epochs = 500  #จำนวนรอบในการ Train
NUM_TRAIN = 2821# จำนวนภาพ Train
NUM_TEST = 125 #จำนวนภาพ Test
dropout_rate = 0.2 

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
import sys
sys.path.append('/media/tohn/SSD/Efficient_USAI/content/efficientnet_keras_transfer_learning')
from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input

# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape) 

# data augmentation เพื่อลดโอกาสการเกิด overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255, # image input 0-255 --> 0-1 เปลี่ยนค่าสี
      rotation_range=40, # หมุนภาพในองศา
      width_shift_range=0.2, #เปลี่ยนความกว้าง
      height_shift_range=0.2, #ปลี่ยนความสูง
      shear_range=0.2, #ทำให้ภาพเบี้ยว
      zoom_range=0.2, #ซุม image มากสุด 20%
      horizontal_flip=True, #พลิกภาพแบบสุ่มตามแนวนอน
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
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(8, activation='softmax', name="fc_out"))        #class --> 8
model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False  

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights)) 

#setting model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
#คำสั่ง Train
history = model.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator, 
      validation_steps= NUM_TEST //batch_size,
      verbose=1,
      use_multiprocessing=True, 
      workers=1) 
 
#save model    
os.makedirs("./models", exist_ok=True)
model.save('./models/Abs_b0_c11_250.h5')

#plot graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
#save plot_acc
plt.savefig('plot_acc_abs.png')

plt.figure()
plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#save plot_loss
plt.savefig('plot_loss_sub.png')

      
      
      




