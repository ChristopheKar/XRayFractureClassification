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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169, DenseNet201
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from losses import binary_focal_loss


# set dataset parameters
# CLASSES = 7
CLASSES = 1
WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 16
# DATASET = 'AUB_WRIST'
DATASET = 'MURA_ALL'
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

def dataset_generator():

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
    model.add(Dense(CLASSES, activation='sigmoid'))

    return model

def fine_tuning(model, conv_base, training_layers):

    conv_base.trainable = True

    for layer in conv_base.layers[:-training_layers]:
        layer.trainable = False

    return model

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def compile_model(model, loss='default', lrate=0.0001):

    adam = Adam(lr=lrate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=True)

    if loss == 'default':
        loss_f = 'binary_crossentropy'
    if loss == 'focal':
        loss_f = [binary_focal_loss(alpha=.25, gamma=2)]

    model.compile(loss=loss_f,
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def fit_model(model, train_gen, val_gen, output_name, log_dir, steps='norm'):

    model_file = os.path.join('models', output_name)
    log_dir = os.path.join('./logs', log_dir)

    # reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=3,
                                  min_delta=0.0001,
                                  verbose=1,
                                  min_lr=0.0000001)

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
                                      callbacks=[checkpoint, tensorboard, reduce_lr])
    elif steps == 'init':
        history = model.fit_generator(train_gen,
                                      steps_per_epoch=100,epochs=25,
                                      validation_data=val_gen,
                                      validation_steps=50,
                                      callbacks=[checkpoint, tensorboard, reduce_lr])

    elif steps == 'fine':

        history = model.fit_generator(train_gen,
                                       steps_per_epoch=150,
                                       epochs=125,
                                       validation_data=val_gen,
                                       validation_steps=50,
                                       callbacks=[checkpoint, tensorboard, reduce_lr])

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

def run_model(backbone, output, logs, loss='default'):

    base_model = backbone(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    model = create_fclayer(base_model)
    train_datagen, validation_datagen = dataset_generator()
    train_generator, validation_generator = dir_generator(train_datagen, validation_datagen)
    model = compile_model(model, loss=loss)
    model.summary()
    from shutil import copyfile
    copyfile(os.path.realpath(__file__), './logs/train.py')
    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'init')
    draw_plots(hist, logs)
    model = fine_tuning(model, base_model, 19)
    model = compile_model(model, loss=loss)
    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'fine')
    draw_plots(hist, logs)

if __name__ == '__main__':

    start_time = time.time()
    run_model(DenseNet169, 'd169_mura_class.h5', 'd169_mura_class', 'default')
    end_time = time.time()
    print('Total time: {:.3f}'.format((end_time - start_time)/3600))
