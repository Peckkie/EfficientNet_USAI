##จัดการ data
import pandas as pd
import numpy as np
import shutil

df = pd.read_csv (r'/home/yupaporn/EfficientNet_USAI/final_training_table.csv')

msk = np.random.rand(len(df)) < 0.9 #split
test = df[~msk]
train= df[msk]

import os
# os.path.abspath('/media/tohn/SSD/Efficient_USAI')
os.chdir('/media/tohn/SSD/Nor_ABnor_Network_24/content/efficientnet_keras_transfer_learning/')
  #choose gpu on processing 
os.environ["CUDA_VISIBLE_DEVICES"]="" # second gpu  

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

##Hyper parameters
batch_size = 50 

width = 150 
height = 150 
input_shape = (height, width, 3) #ขนาด image enter

epochs = 10
NUM_TRAIN = len(Nor_path_train)+len(ABn_path_train)  
NUM_TEST = len(ABn_path_validation)+len(Nor_path_validation) 
dropout_rate = 0.2

import sys
sys.path.append('/media/tohn/SSD/test_func/content/efficientnet_keras_transfer_learning')
from efficientnet import EfficientNetB3 as Net
from efficientnet import center_crop_and_resize, preprocess_input

# data augmentation เพื่อลดโอกาสการเกิด overfitting
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

#Show architecture model
# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape) 

x = conv_base.output  #(in your case pre_trained model is efficientnet-b3)
global_average_layer = GlobalAveragePooling2D()(x)
dropout_layer_1 = Dropout(0.50)(global_average_layer)
prediction_layer = Dense(1, activation='sigmoid')(dropout_layer_1)

model = Model(inputs= pre_trained_model.input, outputs=prediction_layer) 
model.summary()
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.GlobalMaxPooling2D(name="gap"))
# # model.add(layers.Flatten(name="flatten"))
# if dropout_rate > 0:
#     model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# # model.add(layers.Dense(256, activation='relu', name="fc1"))
# model.add(layers.Dense(1, activation='sigmoid', name="fc_out"))        
# model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense

#Training
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
    
    #สร้าง folder TensorBoard
root_logdir = '/media/tohn/SSD/Nor_ABnor_Network_24/my_logs'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(run_logdir)
#ส้รางไฟล์เก็บโมเดล
os.makedirs("./models", exist_ok=True)
#คำสั่ง Train
history = model.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator, 
      validation_steps= NUM_TEST //batch_size,
      verbose=1, 
      use_multiprocessing=True, 
      workers=1,
      callbacks = [tensorboard_cb,callbacks.ModelCheckpoint(filepath='./models/Nor_ABnor_b3_R1_call.h5', save_freq = 'epoch')]) #

hist_df = pd.DataFrame(history.history) 
hist_df.to_csv('hist_df_R1.csv')

#save model   
model.save('./models/Nor_ABnor_b3_R1.h5')

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
plt.savefig('plot_acc_Nor_ABnor_b3_R1.png')

plt.figure()
plt.plot(epochs_x, loss, 'ro', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
    #save plot_loss
plt.savefig('plot_loss_Nor_ABnor_b3_R1.png')

# #Unfreez
# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'multiply_24':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
# print('This is the number of trainable layers '
#       'after freezing the conv base:', len(model.trainable_weights))  

# #Train R2
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
#               metrics=['acc'])

# #สร้าง folder TensorBoard
# root_logdir = '/media/tohn/SSD/Nor_ABnor_Network_24/my_logs_2'
# def get_run_logdir():
#     import time
#     run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
#     return os.path.join(root_logdir,run_id)
# run_logdir = get_run_logdir()

# #คำสั่ง Train
# tensorboard_cb = callbacks.TensorBoard(run_logdir)

# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch= NUM_TRAIN //batch_size,
#       epochs=epochs,
#       validation_data=validation_generator, 
#       validation_steps= NUM_TEST //batch_size,
#       verbose=1, 
#       use_multiprocessing=True, 
#       workers=1,
#       callbacks = [tensorboard_cb,callbacks.ModelCheckpoint(filepath='./models/Nor_ABnor_b3_R2_call.h5', save_freq = 'epoch')]) #

# hist_df = pd.DataFrame(history.history) 
# hist_df.to_csv('hist_df_R2.csv')

# #save model    
# model.save('./models/Nor_ABnor_b3_R2.h5')

# ##plot graph
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_x = range(len(acc))

# plt.plot(epochs_x, acc, 'ro', label='Training acc')
# plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#     #save plot_acc
# plt.savefig('plot_acc_Nor_ABnor_b3_R2.png')

# plt.figure()
# plt.plot(epochs_x, loss, 'ro', label='Training loss')
# plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#     #save plot_loss
# plt.savefig('plot_loss_Nor_ABnor_b3_R2.png')












