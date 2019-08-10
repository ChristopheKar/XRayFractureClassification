### references ###
# https://keras.io/applications/#usage-examples-for-image-classification-models
# https://github.com/keras-team/keras/issues/9214
##################

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.densenet import DenseNet169, DenseNet201
from keras.applications.densenet import preprocess_input as preprocess_dense
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# set dataset parameters
# CLASSES = 7
CLASSES = 1
WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 16
DATASET = 'AUB_WRIST'
# DATASET = 'MURA_ALL'
# DATASET = 'MURA_WRIST'

if DATASET == 'AUB_WRIST':
    TRAIN_DIR = '/home/ubuntu/wrist/datasets/split/train'
    VAL_DIR = '/home/ubuntu/wrist/datasets/split/val'
    NUM_TRAIN = 15220
    NUM_VAL = 1904

if DATASET == 'MURA_ALL':
    TRAIN_DIR = '/home/ubuntu/wrist/datasets/MURA_classification/train'
    VAL_DIR = '/home/ubuntu/wrist/datasets/MURA_classification/valid'
    NUM_TRAIN = 36804
    NUM_VAL = 3197

if DATASET == 'MURA_WRIST':
    TRAIN_DIR = '/home/ubuntu/wrist/datasets/MURA_wrist/train'
    VAL_DIR = '/home/ubuntu/wrist/datasets/MURA_wrist/valid'
    NUM_TRAIN = 9748
    NUM_VAL = 679

# set training parameters
EPOCHS = 100
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
VALIDATION_STEPS = NUM_VAL//BATCH_SIZE

def dataset_generator(preprocess_func):

    # create dataset generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    return train_datagen, validation_datagen

def dir_generator(train_datagen, validation_datagen):

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary')

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary')

    return train_generator, validation_generator

def create_fclayer(conv_base):

    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(CLASSES, activation='softmax'))

    return model

# def fine_tuning(model, conv_base, layer_name):
#
#     conv_base.trainable = True
#     set_trainable = False
#
#     for layer in conv_base.layers:
#
#         if layer.name == layer_name:
#             set_trainable = True
#         if set_trainable:
#             layer.trainable = True
#         else:
#             layer.trainable = False
#
#     return model

def fine_tuning(model, conv_base, training_layers):

    for layer in conv_base.layers[:-training_layers]:
        layer.training = False

    return model

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def compile_model(model, opt='adam'):

    if opt == 'rmsprop':
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    if opt == 'adam':
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001, decay=0.01),
                      metrics=['accuracy'])

    return model

def fit_model(model, train_gen, val_gen, output_name, log_dir, steps='norm'):

    model_file = os.path.join('models', output_name)
    log_dir = os.path.join('./logs', log_dir)

    # save best models
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    # log to tensorboard
    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    # set up learning rate schedule
    lrate = LearningRateScheduler(step_decay)

    # fit model
    if steps == 'norm':
        history = model.fit_generator(train_gen,
                                      epochs=EPOCHS,
                                      steps_per_epoch=STEPS_PER_EPOCH,
                                      validation_data=val_gen,
                                      validation_steps=VALIDATION_STEPS,
                                      callbacks=[checkpoint, tensorboard])
    elif steps == 'init':
        history = model.fit_generator(train_gen,
                                      steps_per_epoch=100,epochs=25,
                                      validation_data=val_gen,
                                      validation_steps=50,
                                      callbacks=[checkpoint, tensorboard])

    elif steps == 'fine':

        history = model.fit_generator(train_gen,
                                       steps_per_epoch=150,
                                       epochs=150,
                                       validation_data=val_gen,
                                       validation_steps=50,
                                       callbacks=[checkpoint, tensorboard])

    return history, model

def draw_plots(hist, logs):

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(os.path.join('./logs', logs, 'acc.png'))
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(os.path.join('./logs', logs, 'loss.png'))

    from shutil import copyfile
    copyfile(os.path.getrealpath(__file__), './logs/train.py')

def run_model(backbone, preprocess_func, output, logs, opt='adam', act='relu'):

    base_model = backbone(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    model = create_fclayer(base_model)
    train_datagen, validation_datagen = dataset_generator(preprocess_func)
    train_generator, validation_generator = dir_generator(train_datagen, validation_datagen)
    model = compile_model(model, opt)
    model.summary()
    from shutil import copyfile
    copyfile(os.path.getrealpath(__file__), './logs/train.py')
    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'init')
    draw_plots(hist, logs)
    model = fine_tuning(model, base_model, 19)
    model = compile_model(model, opt)
    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'fine')
    draw_plots(hist, logs)

if __name__ == '__main__':

    start_time = time.time()

    # run_model(ResNet50, preprocess_resnet, 'resnet50_pets.h5', 'resnet50_pets')
    run_model(DenseNet169, preprocess_dense, 'd169_finetune19.h5', 'd169_finetune19')
    end_time = time.time()
    print('Total time: {:.3f}'.format((end_time - start_time)/3600))
