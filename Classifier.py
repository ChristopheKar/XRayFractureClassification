from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import math
import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt

# import metrics
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# import keras utilities and layers
from keras import backend as K
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau

from losses import binary_focal_loss, categorical_focal_loss

# import keras models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

class ClassifierCNN:

    def __init__(self, backbone, dataset, model_name):

        K.clear_session()

        self.home = os.environ['HOME']

        self.global_root = os.path.join(self.home, 'wrist/classification')
        self.models_root = os.path.join(self.global_root, models)
        self.logs_root = os.path.join(self.global_root, logs)
        self.datasets_root = os.path.join(self.home, 'wrist/datasets')

        self.model_name = model_name + '.h5'
        self.logs_name = model_name

        self.model_path = os.path.join(self.models_root, self.model_name)
        self.logs_path = os.path.join(self.logs_root, self.logs_name)

        self.backbone = backbone
        self.define_dataset(dataset)

        if self.classes == 1:
            self.class_mode = 'binary'
            self.activation = 'sigmoid'
        elif self.classes > 1:
            self.class_mode = 'categorical'
            self.activation = 'softmax'

        self.height = 224
        self.width = 224
        self.batch_size = 2
        self.loss = 'default'
        self.lrate=0.0001

    def define_dataset(self, dataset):

        if dataset == 'AUB_WRIST':
            aub_wrist = os.path.join(self.datasets_root, 'split')
            self.train_dir = os.path.join(aub_wrist, 'train')
            self.val_dir = os.path.join(aub_wrist, 'val')
            self.num_train = 15220
            self.num_val = 1904
            self.classes = 1


        if dataset == 'MURA_ALL':
            mura_all = os.path.join(self.datasets_root, 'MURA_classification')
            self.train_dir = os.path.join(mura_all, 'train')
            self.val_dir = os.path.join(mura_all, 'val')
            self.num_train = 36804
            self.num_val = 3197
            self.classes = 7


        if dataset == 'MURA_WRIST':
            mura_wrist = os.path.join(self.datasets_root, 'MURA_wrist')
            self.train_dir = os.path.join(mura_wrist, 'train')
            self.val_dir = os.path.join(mura_wrist, 'valid')
            self.num_train = 9748
            self.num_val = 679
            self.classes = 1

        if dataset == 'MURA_HUMERUS':
            mura_humerus = os.path.join(self.datasets_root, 'MURA_humerus')
            self.train_dir = os.path.join(mura_humerus, 'train')
            self.val_dir = os.path.join(mura_humerus, 'valid')
            self.num_train = 1272
            self.num_val = 288
            self.classes = 1

    def load_dataset_generator(self):

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
            rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = self.class_mode)

        self.validation_generator = validation_datagen.flow_from_directory(
            self.val_dir,
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = self.class_mode)

    def create_fclayer(self, conv_base, pre=False):

        conv_base.trainable = False

        model = Sequential()
        if pre is False:
            model.add(conv_base)
        else:
            model.add(conv_base.layers[0])
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(CLASSES, activation=self.activation))

        return model

    def fine_tuning(self, model, conv_base, training_layers):

        conv_base.trainable = True

        for layer in conv_base.layers[:-training_layers]:
            layer.trainable = False

        return model

    def step_decay(self, epoch):

        min_lrate = 0.00000001
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        if lrate < min_lrate:
            lrate = min_lrate
        return lrate

    def compile_model():

        adam = Adam(lr=self.lrate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=True)

        if self.loss == 'default' and self.classes == 1:
            loss_f = 'binary_crossentropy'
        if self.loss == 'default' and self.classes > 1:
            loss_f = 'categorical_crossentropy'
        if self.loss == 'focal' and self.classes == 1:
            loss_f = [binary_focal_loss(alpha=.25, gamma=2)]
        if self.loss == 'focal' and self.classes > 1:
            loss_f = [categorical_focal_loss(alpha=.25, gamma=2)]

        self.model.compile(loss=loss_f,
                           optimizer=adam,
                           metrics=['accuracy'])

        return model

    def fit_model(self, steps='init'):

        # reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=3,
                                      min_delta=0.0001,
                                      verbose=1,
                                      min_lr=0.0000001)

        # save best models
        checkpoint = ModelCheckpoint(self.model_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        # log to tensorboard
        tensorboard = TensorBoard(log_dir=self.logs_path,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        # set up learning rate schedule
        lrate = LearningRateScheduler(self.step_decay)

        # Early Stopping on Plateau
        es = EarlyStopping(monitor='val_acc',
                           mode='max',
                           verbose=1,
                           patience=20)

        # fit model
        if steps == 'init':
            self.history = self.model.fit_generator(
                                self.train_generator,
                                steps_per_epoch=100,
                                epochs=25,
                                validation_data=self.validation_generator,
                                validation_steps=50,
                                callbacks=[checkpoint, reduce_lr])

        elif steps == 'fine':

            self.history = self.model.fit_generator(
                                self.train_generator,
                                steps_per_epoch=150,
                                epochs=125,
                                validation_data=self.validation_generator,
                                validation_steps=50,
                                callbacks=[checkpoint, tensorboard, reduce_lr])

    def train(self):

        # creating model
        if isinstance(self.backbone, str):
            model_path = os.path.join(self.models_root, self.backbone)
            base_model = load_model(model_path)
            for i in range(6):
                base_model._layers.pop()
            base_model.summary()
            self.model = create_fclayer(base_model, True)
        else:
            base_model = self.backbone(include_top=False,
                                       input_shape = (self.height,self.width,3),
                                       weights='imagenet')
            self.model = self.create_fclayer(base_model)

        self.load_dataset_generators()
        self.compile_model()
        start_time = time.time()
        self.fit_model('init')
        checkpoint_1 = time.time()
        self.fine_tune(base_model, layers=19)
        self.compile_model()
        checkpoint_2 = time.time()
        self.fit_model('fine')
        end_time = time.time()
        total_time = (checkpoint_1 - start_time) + (end_time - checkpoint_2)
        total_time = total_time / 3600
        # evaluate(model, VAL_DIR, NUM_VAL, logs, hist, total_time)