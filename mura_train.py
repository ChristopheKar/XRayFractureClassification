### references ###
# https://keras.io/applications/#usage-examples-for-image-classification-models
# https://github.com/keras-team/keras/issues/9214
##################

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
keras = tf.keras

from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input as preprocess_dense
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# set dataset parameters
CLASSES = 2
WIDTH, HEIGHT = 200,200
BATCH_SIZE = 16
TRAIN_DIR = '/home/ubuntu/wrist/MURA-v1.1/data/train'
VAL_DIR = '/home/ubuntu/wrist/MURA-v1.1/data/val'
NUM_TRAIN = 9748
NUM_VAL = 659

# set training parameters
EPOCHS = 100
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
VALIDATION_STEPS = NUM_VAL//BATCH_SIZE

def create_fclayer(base_model, activation='relu'):

    # Create classification fully connected layer
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool2d')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation=activation)(x)

    return predictions

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

def get_pre_VGG16(conv_base):

    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # bylo 256
    model.add(Dense(1, activation='sigmoid'))

    return model

def compile_model(base_model, predictions, opt='adam'):

    model = Model(inputs=base_model.input, outputs=predictions)
    #
    # for layer in base_model.layers:
    #     layer.trainable = False

    if opt == 'rmsprop':
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    if opt == 'adam':
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.00001, decay=0.01),
                      metrics=['accuracy'])

    return model

def fit_model(model, train_gen, val_gen, output_name, log_dir):

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
    # fit model
    history = model.fit_generator(
                                train_gen,
                                epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_data=val_gen,
                                validation_steps=VALIDATION_STEPS,
                                callbacks=[checkpoint, tensorboard])

    return history

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

def run_model(backbone, preprocess_func, output, logs, opt='adam', act='relu'):

    base_model = backbone(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    predictions = create_fclayer(base_model, act)
    # predictions = get_pre_VGG16(base_model)
    train_datagen, validation_datagen = dataset_generator(preprocess_func)
    train_generator, validation_generator = dir_generator(train_datagen, validation_datagen)
    model = compile_model(base_model, predictions, opt)

    hist = fit_model(model, train_generator, validation_generator, output, logs)
    draw_plots(hist, logs)


if __name__ == '__main__':

    # run_model(ResNet50, preprocess_resnet, 'resnet50_pets.h5', 'resnet50_pets')
    run_model(VGG16, preprocess_vgg, 'vgg_mura.h5', 'vgg_mura')
