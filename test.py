import pandas as pd
import shutil

df = pd.read_csv (r'/home/yupaporn/codes/USAI/detail1_250.csv')
#manage dataframe
df1 = df
df01 = df1.replace({'Sub Position': {'P11':'P1','P12':'P1','P4':'P41','P43':'P41','P6':'P61','P7':'P71'}})

import os
# os.path.abspath('/media/tohn/SSD/Efficient_USAI')
os.chdir('/media/tohn/SSD/Sub_Efficient_USAI/content/efficientnet_keras_transfer_learning/')

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
validation_P31_dir = os.path.join(validation_dir, 'P31')
os.makedirs(validation_P31_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P32_dir = os.path.join(validation_dir, 'P32')
os.makedirs(validation_P32_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P41_dir = os.path.join(validation_dir, 'P41')
os.makedirs(validation_P41_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P42_dir = os.path.join(validation_dir, 'P42')
os.makedirs(validation_P42_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P51_dir = os.path.join(validation_dir, 'P51')
os.makedirs(validation_P51_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P52_dir = os.path.join(validation_dir, 'P52')
os.makedirs(validation_P52_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P61_dir = os.path.join(validation_dir, 'P61')
os.makedirs(validation_P61_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P62_dir = os.path.join(validation_dir, 'P62')
os.makedirs(validation_P62_dir, exist_ok=True)
validation_P71_dir = os.path.join(validation_dir, 'P71')
os.makedirs(validation_P71_dir, exist_ok=True)
# Directory with our training cat pictures
validation_P72_dir = os.path.join(validation_dir, 'P72')
os.makedirs(validation_P72_dir, exist_ok=True)
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
train_P31_dir = os.path.join(train_dir, 'P31')
os.makedirs(train_P31_dir, exist_ok=True)
# Directory with our training cat pictures
train_P32_dir = os.path.join(train_dir, 'P32')
os.makedirs(train_P32_dir, exist_ok=True)
# Directory with our training cat pictures
train_P41_dir = os.path.join(train_dir, 'P41')
os.makedirs(train_P41_dir, exist_ok=True)
# Directory with our training cat pictures
train_P42_dir = os.path.join(train_dir, 'P42')
os.makedirs(train_P42_dir, exist_ok=True)
# Directory with our training cat pictures
train_P51_dir = os.path.join(train_dir, 'P51')
os.makedirs(train_P51_dir, exist_ok=True)
# Directory with our training cat pictures
train_P52_dir = os.path.join(train_dir, 'P52')
os.makedirs(train_P52_dir, exist_ok=True)
# Directory with our training cat pictures
train_P61_dir = os.path.join(train_dir, 'P61')
os.makedirs(train_P61_dir, exist_ok=True)
# Directory with our training cat pictures
train_P62_dir = os.path.join(train_dir, 'P62')
os.makedirs(train_P62_dir, exist_ok=True)
train_P71_dir = os.path.join(train_dir, 'P71')
os.makedirs(train_P71_dir, exist_ok=True)
# Directory with our training cat pictures
train_P72_dir = os.path.join(train_dir, 'P72')
os.makedirs(train_P72_dir, exist_ok=True)
# Directory with our training cat pictures
train_P8_dir = os.path.join(train_dir, 'P8')
os.makedirs(train_P8_dir, exist_ok=True)

#cut out path 'None'
df1 =df01[df01['Path Crop'] != 'None']

#Create Path images
val = df1[df1['Case'].between(1, 10)]
train = df1[df1['Case'].between(11, 250)]

#Path images of validation
p1_val = val[val['Sub Position']=='P1' ]
p1_path_val = p1_val['Path Crop'].tolist() 
p2_val = val[val['Sub Position']=='P2' ]
p2_path_val = p2_val['Path Crop'].tolist() 
p31_val = val[val['Sub Position']=='P31' ]
p31_path_val = p31_val['Path Crop'].tolist() 
p32_val = val[val['Sub Position']=='P32' ]
p32_path_val = p32_val['Path Crop'].tolist()
p41_val = val[val['Sub Position']=='P41' ]
p41_path_val = p41_val['Path Crop'].tolist()
p42_val = val[val['Sub Position']=='P42' ]
p42_path_val = p42_val['Path Crop'].tolist() 
p51_val = val[val['Sub Position']=='P51' ]
p51_path_val = p51_val['Path Crop'].tolist()
p52_val = val[val['Sub Position']=='P52' ]
p52_path_val = p52_val['Path Crop'].tolist() 
p61_val = val[val['Sub Position']=='P61' ]
p61_path_val = p61_val['Path Crop'].tolist()
p62_val = val[val['Sub Position']=='P62' ]
p62_path_val = p62_val['Path Crop'].tolist() 
p71_val = val[val['Sub Position']=='P71' ]
p71_path_val = p71_val['Path Crop'].tolist() 
p72_val = val[val['Sub Position']=='P72' ]
p72_path_val = p72_val['Path Crop'].tolist() 
p8_val = val[val['Sub Position']=='P8' ]
p8_path_val = p8_val['Path Crop'].tolist() 

#Path images of train
p1_train = train[train['Sub Position']=='P1' ]
p1_path_train = p1_train['Path Crop'].tolist() 
p2_train = train[train['Sub Position']=='P2' ]
p2_path_train = p2_train['Path Crop'].tolist() 
p31_train = train[train['Sub Position']=='P31' ]
p31_path_train = p31_train['Path Crop'].tolist()
p32_train = train[train['Sub Position']=='P32' ]
p32_path_train = p32_train['Path Crop'].tolist()
p41_train = train[train['Sub Position']=='P41' ]
p41_path_train = p41_train['Path Crop'].tolist()
p42_train = train[train['Sub Position']=='P42' ]
p42_path_train = p42_train['Path Crop'].tolist()
p51_train = train[train['Sub Position']=='P51' ]
p51_path_train = p51_train['Path Crop'].tolist()
p52_train = train[train['Sub Position']=='P52' ]
p52_path_train = p52_train['Path Crop'].tolist() 
p61_train = train[train['Sub Position']=='P61' ]
p61_path_train = p61_train['Path Crop'].tolist()
p62_train = train[train['Sub Position']=='P62' ]
p62_path_train = p62_train['Path Crop'].tolist()
p71_train = train[train['Sub Position']=='P71' ]
p71_path_train = p71_train['Path Crop'].tolist()
p72_train = train[train['Sub Position']=='P72' ]
p72_path_train = p72_train['Path Crop'].tolist()
p8_train = train[train['Sub Position']=='P8' ]
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
    
fnames = p31_path_val  
for fname in fnames:
    dst = os.path.join(validation_P31_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p32_path_val  
for fname in fnames:
    dst = os.path.join(validation_P32_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p41_path_val  
for fname in fnames:
    dst = os.path.join(validation_P41_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)  

fnames = p42_path_val  
for fname in fnames:
    dst = os.path.join(validation_P42_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)  
    
fnames = p51_path_val  
for fname in fnames:
    dst = os.path.join(validation_P51_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p52_path_val  
for fname in fnames:
    dst = os.path.join(validation_P52_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p61_path_val  
for fname in fnames:
    dst = os.path.join(validation_P61_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p62_path_val  
for fname in fnames:
    dst = os.path.join(validation_P62_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p71_path_val  
for fname in fnames:
    dst = os.path.join(validation_P71_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p72_path_val  
for fname in fnames:
    dst = os.path.join(validation_P72_dir, os.path.basename(fname))
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
    
fnames = p31_path_train  
for fname in fnames:
    dst = os.path.join(train_P31_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p32_path_train  
for fname in fnames:
    dst = os.path.join(train_P32_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p41_path_train  
for fname in fnames:
    dst = os.path.join(train_P41_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p42_path_train  
for fname in fnames:
    dst = os.path.join(train_P42_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p51_path_train  
for fname in fnames:
    dst = os.path.join(train_P51_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

fnames = p52_path_train  
for fname in fnames:
    dst = os.path.join(train_P52_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p61_path_train  
for fname in fnames:
    dst = os.path.join(train_P61_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p62_path_train  
for fname in fnames:
    dst = os.path.join(train_P62_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p71_path_train  
for fname in fnames:
    dst = os.path.join(train_P71_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
     
fnames = p72_path_train  
for fname in fnames:
    dst = os.path.join(train_P72_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
    
fnames = p8_path_train  
for fname in fnames:
    dst = os.path.join(train_P8_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

#Show number of train&val

print('total training images:', len(os.listdir(train_P1_dir))+len(os.listdir(train_P2_dir))+len(os.listdir(train_P31_dir))+len(os.listdir(train_P32_dir))
      +len(os.listdir(train_P41_dir))+len(os.listdir(train_P42_dir))+len(os.listdir(train_P51_dir))+len(os.listdir(train_P52_dir))+len(os.listdir(train_P61_dir))
      +len(os.listdir(train_P62_dir))+len(os.listdir(train_P71_dir))+len(os.listdir(train_P72_dir))+len(os.listdir(train_P8_dir)),'\n')

print('total training images:', len(os.listdir(validation_P1_dir))+len(os.listdir(validation_P2_dir))+len(os.listdir(validation_P31_dir))
      +len(os.listdir(validation_P32_dir))+len(os.listdir(validation_P41_dir))+len(os.listdir(validation_P42_dir))+len(os.listdir(validation_P51_dir))
      +len(os.listdir(validation_P52_dir))+len(os.listdir(validation_P61_dir))+len(os.listdir(validation_P62_dir))+len(os.listdir(validation_P71_dir))
      +len(os.listdir(validation_P72_dir))+len(os.listdir(validation_P8_dir)))
   
  #choose gpu on processing 
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu  

#import Library
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
import os
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

import sys
sys.path.append('/media/tohn/SSD/Sub_Efficient_USAI/content/efficientnet_keras_transfer_learning')
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

#Show architecture model
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(13, activation='softmax', name="fc_out"))        #class --> 13 **Sub
model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense

#Training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

#สร้าง folder TensorBoard
root_logdir = '/media/tohn/SSD/Sub_Efficient_USAI/my_logs'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

#คำสั่ง Train
tensorboard_cb = callbacks.TensorBoard(run_logdir)
history = model.fit_generator(
      train_generator,#โหลดdataเข้ามา
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator, #validation_data=(x_valid, y_valid): ใส่ data ที่เราแยกไว้เพื่อดูผล Model ว่าเกิด Overfitting เริ่มที่จุดใด
      validation_steps= NUM_TEST //batch_size,
      verbose=1, #โชว์ผลลัพธ์ 0:ปิด
      use_multiprocessing=True, #ใช้ GPU หลายตัว
      workers=1,
      callbacks = tensorboard_cb) #ทำพร้อมกันที่ละ 4 ตัว

#save model    
os.makedirs("./models", exist_ok=True)
model.save('./models/Sub_b0_c11_250.h5')

#plot graph
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
plt.savefig('plot_acc_sub.png')

plt.figure()
plt.plot(epochs_x, loss, 'ro', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#save plot_loss
plt.savefig('plot_loss_sub.png')



















