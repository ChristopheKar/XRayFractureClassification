from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import math
import numpy as np
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

from Classifier import ClassifierCNN

if __name__ == '__main__':

    # cls = ClassifierCNN(DenseNet169, 'MURA_HUMERUS', 'd169_mura_humerus_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'MURA_HUMERUS', 'd169_mura_humerus_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'MURA_HUMERUS', 'd169_mura_humerus_224_v3')
    # cls.layers = 224
    # cls.train()
    #
    # cls = ClassifierCNN(DenseNet169, 'MURA_WRIST', 'd169_mura_wrist_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'MURA_WRIST', 'd169_mura_wrist_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'MURA_WRIST', 'd169_mura_wrist_224_v3')
    # cls.layers = 224
    # cls.train()
    #
    # cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v3')
    # cls.layers = 224
    # cls.train()

    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_19_v1')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_19_v1')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v1')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v1')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_19_v2')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_19_v2')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v2')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v2')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_19_v3')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_19_v3')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v3')
    # cls.layers = 19
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_HUMERUS', 'd169_mura_humerus_224_224_v3')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_224_v3')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v3')
    # cls.layers = 224
    # cls.train()

    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_WRIST', 'd169_mura_wrist_224_224_v4')
    # cls.layers = 224
    # cls.train()
    #
    # cls = ClassifierCNN(DenseNet169, 'MURA_ALL', 'd169_mura_class_aug_224')
    # cls.layers = 224
    # cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v4')
    cls.layers = 19
    cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v4')
    cls.layers = 224
    cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v5')
    cls.layers = 19
    cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v5')
    cls.layers = 224
    cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v6')
    cls.layers = 19
    cls.train()

    cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v6')
    cls.layers = 224
    cls.train()
