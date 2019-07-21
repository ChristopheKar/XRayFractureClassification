#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:41:11 2018

@author: michal
"""


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.applications import VGG16


# =============================================================================
#                                   VGG16
# =============================================================================


def get_my_VGG16():
    
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),activation="relu",padding="same",input_shape=(150,150,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
        
    model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
        
        
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# =============================================================================
#                                 VGG16 pretrained
# =============================================================================

def get_pre_VGG16(conv_base):
    

    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # bylo 256
    model.add(Dense(1, activation='sigmoid'))

    return model

# =============================================================================
#                              Resnet50 pretrained
# =============================================================================

def get_pre_Resnet50(res_base):
    
    res_base.trainable = False

    model = Sequential()
    model.add(res_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # bylo 256
    model.add(Dense(1, activation='sigmoid'))

    return model

# =============================================================================
#                              Fine Tuning
# =============================================================================

def fine_tuning(model,conv_base,layer_name):
    
    conv_base.trainable = True
    set_trainable = False
    
    for layer in conv_base.layers:
        
        if layer.name == layer_name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    return model

# =============================================================================
#                               Test Model
# =============================================================================

def test_model(model,test_datagen,test_directory):
    
    model_stack = {}
    
    test_generator = test_datagen.flow_from_directory(test_directory,
                                                      target_size=(200, 200),
                                                      batch_size=32,class_mode='binary')

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

    model_stack["test_acc"] = test_acc
    model_stack["test_loss"] = test_loss

    return model_stack


