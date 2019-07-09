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

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
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
CLASSES = 1
WIDTH, HEIGHT = 300,300
BATCH_SIZE = 16
TRAIN_DIR = '/home/ubuntu/wrist/PetImages'
VAL_DIR = '/home/ubuntu/wrist/PetImages'
NUM_TRAIN = 24998
NUM_VAL = 24998

# TRAIN_CSV = '/home/user/chris/datasets/MURA-v1.1/train_image_paths.csv'
# VALID_CSV = '/home/user/chris/datasets/MURA-v1.1/valid_image_paths.csv'
# NUM_TRAIN = 36808
# NUM_VAL = 3197

# set training parameters
EPOCHS = 100
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
VALIDATION_STEPS = NUM_VAL//BATCH_SIZE

def create_fclayer(base_model, activation='relu'):

    # Create classification fully connected layer
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool2d')(x)
    x = Dropout(0.4)(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation=activation)(x)
    # x = base_model.output
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(CLASSES, activation='softmax')(x)
    return predictions

def dataset_generator(preprocess_func):

    # create dataset generators
    #train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    #validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
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

    # create batches
    x_batch, y_batch = next(train_generator)

    return train_generator, validation_generator

def csv_generator(df_train, df_val, train_datagen, validation_datagen):

    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        directory=None,
        x_col='path',
        y_col='label',
        has_ext=True,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary')

    validation_generator = validation_datagen.flow_from_dataframe(
        df_val,
        directory=None,
        x_col='path',
        y_col='label',
        has_ext=True,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary')

    return train_generator, validation_generator

def compile_model(base_model, predictions, opt='adam'):

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # for layer in base_model.layers:
        # layer.trainable = True

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

    df_train = pd.read_csv(TRAIN_CSV, header=None)
    df_train = df_train.rename(index=str, columns = {0:"path"})
    df_train['label'] = 'none'
    df_train.loc[df_train['path'].str.contains("positive"), 'label'] = 'abnormal'
    df_train.loc[df_train['path'].str.contains("negative"), 'label'] = 'normal'
    df_train['path'] = '/home/user/chris/datasets/' + df_train['path'].astype(str)

    df_val = pd.read_csv(VALID_CSV, header=None)
    df_val = df_val.rename(index=str, columns = {0:"path"})
    df_val['label'] = 'none'
    df_val.loc[df_val['path'].str.contains("positive"), 'label'] = 'abnormal'
    df_val.loc[df_val['path'].str.contains("negative"), 'label'] = 'normal'
    df_val['path'] = '/home/user/chris/datasets/' + df_val['path'].astype(str)

    base_model = backbone(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    predictions = create_fclayer(base_model, act)
    train_datagen, validation_datagen = dataset_generator(preprocess_func)
    train_generator, validation_generator = dir_generator(train_datagen, validation_datagen)
    #train_generator, validation_generator = csv_generator(df_train, df_val, train_datagen, validation_datagen)
    model = compile_model(base_model, predictions, opt)

    hist = fit_model(model, train_generator, validation_generator, output, logs)
    draw_plots(hist, logs)


if __name__ == '__main__':

    # run_model(ResNet50, preprocess_resnet, 'resnet50_pets.h5', 'resnet50_pets')
    run_model(InceptionV3, preprocess_inception, 'iv3_pets.h5', 'iv3_pets')
