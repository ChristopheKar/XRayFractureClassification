from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, Adadelta
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
if 'user' in os.environ['HOME']:
    TRAIN_DIR = '/home/user/chris/datasets/wrist_fyp/split/train'
    VAL_DIR = '/home/user/chris/datasets/wrist_fyp/split/val'
else:
    # # AUB WRIST
    # TRAIN_DIR = '/home/ubuntu/wrist/datasets/split/train'
    # VAL_DIR = '/home/ubuntu/wrist/datasets/split/val'
    # # MURA WRIST
    # TRAIN_DIR = '/home/ubuntu/wrist/datasets/MURA_wrist/train'
    # VAL_DIR = '/home/ubuntu/wrist/datasets/MURA_wrist/valid'
    # # MURA ALL
    # TRAIN_DIR = '/home/ubuntu/wrist/datasets/MURA_classification/train'
    # VAL_DIR = '/home/ubuntu/wrist/datasets/MURA_classification/valid'
    # PETS
    TRAIN_DIR = '/home/ubuntu/wrist/datasets/PetImages/train'
    VAL_DIR = '/home/ubuntu/wrist/datasets/PetImages/valid'


# # AUB FYP SPLIT
# NUM_TRAIN = 15220
# NUM_VAL = 1904

# # MURA
# NUM_TRAIN = 36804
# NUM_VAL = 3197

# # MURA WRIST
# NUM_TRAIN = 9748
# NUM_VAL = 679

# PETS
NUM_TRAIN = 24946
NUM_VAL = 400

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
        class_mode = 'input')

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        target_size = (HEIGHT, WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'input')

    return train_generator, validation_generator

def create_fclayer(conv_base):

    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_dec'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_dec'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_dec'))
    model.add(UpSampling2D((2, 2), name='block5_pool_dec'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_dec'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2_dec'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3_dec'))
    model.add(UpSampling2D((2, 2), name='block3_pool_dec'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_dec'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_dec'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_dec'))
    model.add(UpSampling2D((2, 2), name='block2_pool_dec'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_dec'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_dec'))
    model.add(UpSampling2D((2, 2), name='block1_pool_dec'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_dec'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_dec'))
    model.add(UpSampling2D((2, 2), name='block6_pool_dec'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same', name='block6_conv1_dec'))

    return model

def fine_tuning(model, conv_base, layer_name):

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

    if opt == 'autoenc':
        model.compile(optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                      loss='binary_crossentropy')

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
                                      steps_per_epoch=100,epochs=125,
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

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(os.path.join('./logs', logs, 'loss.png'))

def run_model(backbone, output, logs, opt='adam', act='relu'):

    base_model = backbone(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    model = create_fclayer(base_model)
    train_datagen, validation_datagen = dataset_generator()
    train_generator, validation_generator = dir_generator(train_datagen, validation_datagen)
    model = compile_model(model, opt)
    model.summary()

    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'init')
    draw_plots(hist, logs)
    model = fine_tuning(model, base_model, 'block5_conv1')
    model = compile_model(model, opt)
    hist, model = fit_model(model, train_generator, validation_generator, output, logs, 'fine')
    draw_plots(hist, logs)

if __name__ == '__main__':

    start_time = time.time()
    run_model(VGG16, 'vgg_pets_autoenc.h5', 'vgg_pets_autoenc', opt='autoenc')
    end_time = time.time()
    print('Total time: {:.3f}'.format((end_time - start_time)/3600))


# input_shape = (224, 224, 3)
#
# img_input = Input(shape=input_shape)
#
# # Encoder Network
#
# # Block 1
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#
# # Block 2
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#
# # Block 3
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#
# # Block 4
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
# # Block 5
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#
# # Decoder Network
#
# # Block 1
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_dec')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_dec')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_dec')(x)
#
# # Block 2
# x = UpSampling2D((2, 2), name='block5_pool_dec')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1_dec')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2_dec')(x)
# x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3_dec')(x)
#
# # Block 3
# x = UpSampling2D((2, 2), name='block3_pool_dec')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_dec')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_dec')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_dec')(x)
#
# # Block 4
# x = UpSampling2D((2, 2), name='block2_pool_dec')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_dec')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_dec')(x)
#
# # Block 5
# x = UpSampling2D((2, 2), name='block1_pool_dec')(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_dec')(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_dec')(x)
#
# x = UpSampling2D((2, 2), name='block6_pool_dec')(x)
# x = Conv2D(3, (3, 3), activation='relu', padding='same', name='block6_conv1_dec')(x)
#
# autoencoder = Model(img_input, x)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.summary()
