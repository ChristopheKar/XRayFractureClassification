# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

# =============================================================================
#                                 Libraries
# =============================================================================
import os
import shutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from modules import my_utils
from modules import models
from modules import plots

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import optimizers
from keras.preprocessing import image

# =============================================================================
#                           Create folder structure
# =============================================================================

from os.path import join
HOME = os.environ['HOME']

main_folder_path = join(HOME, 'wrist/classification/MURA/MURA-v1.1')
dataset_path = join(HOME, 'wrist/MURA-v1.1')
model_path = join(HOME, 'wrist/classification/MURA/MURA-v1.1/models')
plots_path = join(HOME, 'wrist/classification/MURA/MURA-v1.1/plots')


os.chdir(main_folder_path) # Go to main directory

#Get hand XRAY data
my_utils.copy_data(os.path.join(dataset_path,"train/XR_HAND"),os.path.join(main_folder_path,"train/XR_HAND"))
my_utils.copy_data(os.path.join(dataset_path,"valid/XR_HAND"),os.path.join(main_folder_path,"valid/XR_HAND"))

# Get csv files
my_utils.create_folder(os.path.join(main_folder_path,"CSV"))
my_utils.get_csv(dataset_path,os.path.join(main_folder_path,"CSV"))

# Data work folder
hand_path = os.path.join(main_folder_path,"hand")
my_utils.create_folder(hand_path)

# Train
hand_train = os.path.join(hand_path,"train")
my_utils.create_folder(hand_train)

hand_train_normal = os.path.join(hand_train,"normal")
my_utils.create_folder(hand_train_normal)

hand_train_abnormal = os.path.join(hand_train,"abnormal")
my_utils.create_folder(hand_train_abnormal)

# Valid
hand_valid = os.path.join(hand_path,"valid")
my_utils.create_folder(hand_valid)

hand_valid_normal = os.path.join(hand_valid,"normal")
my_utils.create_folder(hand_valid_normal)

hand_valid_abnormal = os.path.join(hand_valid,"abnormal")
my_utils.create_folder(hand_valid_abnormal)



# =============================================================================
#                               Preprocessing
# =============================================================================

#CSV
main_structure = my_utils.get_main_f(os.listdir())
os.chdir(main_structure[5])
csv_structure = my_utils.read_csv(os.listdir())
path_train,path_valid,label_valid,label_train = my_utils.read_csv(os.listdir())
os.chdir(main_folder_path)

# GET HAND XRAY

#Change columns name
path_train = path_train.rename(columns = {'MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png':'img'})
path_valid = path_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png':'img'})

# Get only HAND directories
path_train = my_utils.get_selected_feature(path_train,"MURA-v1.1/train/XR_HAND")
path_valid = my_utils.get_selected_feature(path_valid,"MURA-v1.1/valid/XR_HAND")

# Get radiologist predictions
label_train = label_train.rename(columns = {"MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/":'img'})
label_valid = label_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/':'img'})

label_train = my_utils.get_selected_feature(label_train,"XR_HAND")
label_valid = my_utils.get_selected_feature(label_valid,"XR_HAND")

# Get normal/abnormal distribution
distr_hand_train = my_utils.get_data_distribution(path_train)
distr_hand_valid = my_utils.get_data_distribution(path_valid)


# Data separation

# 1.Rename pathes
path_train = my_utils.re_path(path_train)
path_valid = my_utils.re_path(path_valid)

# 2. Separate images to folder classes

os.path.exists("/home/michal/Pulpit/MURA/MURA-v1.1/valid/XR_HAND/patient00008/study1_positive")
os.path.exists("/home/michal/Pulpit/MURA/MURA-v1.1/valid/XR_HAND/patient00008/study1_positive")

my_utils.separate_images(path_train,hand_train_abnormal,hand_train_normal)
my_utils.separate_images(path_valid,hand_valid_abnormal,hand_valid_normal)




# =============================================================================
#                               Data Overview
# =============================================================================

#Get sample image
img_path = ('/home/michal/Pulpit/MURA/MURA-v1.1/valid/XR_HAND/patient11497/study1_positive/image3.png')
img = image.load_img(img_path)
img1 = image.img_to_array(img)

img
print("\nSummary of hand images : \n")

print("Image height : ", img1.shape[0])
print("Image width : ", img1.shape[1])
print("Image depth : ", img1.shape[2])

print("\nTrain images : ",len(path_train))
print("Valid images : ",len(path_valid))

print("Train abnormal img : ",len(os.listdir(hand_train_abnormal)))
print("Train normal img : ",len(os.listdir(hand_train_normal)))
print("Valid abnormal img : ",len(os.listdir(hand_valid_abnormal)))
print("Valid normal img : ",len(os.listdir(hand_valid_normal)))



# =============================================================================
#                               Generator settings
# =============================================================================

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        hand_train,
        target_size=(200, 200),
        batch_size=16,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        hand_valid,
        target_size=(200, 200),
        batch_size=16,
        class_mode='binary')


# =============================================================================
#                                 VGG16
# =============================================================================

models_stack = {}
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(200, 200, 3))

model = models.get_pre_VGG16(conv_base)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5), #
              metrics=['acc'])

model.summary()

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,epochs=25,
                              validation_data=validation_generator,
                              validation_steps=50,verbose=2)


models_stack["VGG16"] = models.test_model(model,test_datagen,hand_valid)
models.save_model(model,model_path,"VGG16.h5")



# =============================================================================
#                               VGG16 fine_tune
# =============================================================================

model = models.fine_tuning(model,conv_base,'block5_conv1')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

model.summary()
history1 = model.fit_generator(train_generator,
                               steps_per_epoch=125,
                               epochs=125,
                               validation_data=validation_generator,
                               validation_steps=50)

plots.plot_training(history1,plots_path,"VGG16_fine_tune")
models_stack["VGG16_fine_tune"] = models.test_model(model,test_datagen,hand_valid)


models.save_model(model,model_path,"VGG16_fine_tune.h5")

# =============================================================================
#                               RESNET50
# =============================================================================

from keras.applications import ResNet50

res_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(200, 200, 3))

resnet = models.get_pre_Resnet50(res_base)
resnet.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), #
              metrics=['acc'])

resnet.summary()

history2 = resnet.fit_generator(train_generator,
                              steps_per_epoch=100,epochs=10,
                              validation_data=validation_generator,
                              validation_steps=50,verbose=2)

models_stack["Resnet"] = models.test_model(resnet,test_datagen,hand_valid)

models.save_model(model,model_path,"RestNet.h5")

# =============================================================================
#                               ResNet fine_tune
# =============================================================================

resnet = models.fine_tuning(resnet,res_base,'res5b_branch2c')
resnet.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

resnet.summary()
history1 = resnet.fit_generator(train_generator,
                               steps_per_epoch=200,
                               epochs=125,
                               validation_data=validation_generator,
                               validation_steps=50)

models_stack["RestNet_fine_tune"] = models.test_model(resnet,test_datagen,hand_valid)

# =============================================================================
#                                   Test 1 part
# =============================================================================



test_models_by_batch = {}
batch_sizes = [2,4,8,16,32,64,128]

for i in batch_sizes:

    train_generator = train_datagen.flow_from_directory(
    hand_train,
    target_size=(200, 200),
    batch_size=i,
    class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
    hand_valid,
    target_size=(200, 200),
    batch_size=i,
    class_mode='binary')

    test_models_by_batch[str(i)] = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2)

for i in batch_sizes:
    x = test_models_by_batch[str(i)]
    x = x.history["val_acc"]
    plt.plot(x,label ="Batch: " + str( i))
    plt.legend()
    plt.title("Batch size metrics comparison",fontsize = 26)
    plt.ylabel("val_acc")
    plt.xlabel("epochs")

for i in batch_sizes:
    x = test_models_by_batch[str(i)]
    x = x.history["acc"]
    plt.plot(x,label ="Batch: " + str( i))
    plt.legend()
    plt.title("Batch size metrics comparison",fontsize = 26)
    plt.ylabel("acc")
    plt.xlabel("epochs")
