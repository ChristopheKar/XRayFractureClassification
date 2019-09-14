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

    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v4')
    # cls.layers = 19
    # cls.train()
    #
    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v4')
    # cls.layers = 224
    # cls.train()
    #
    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v5')
    # cls.layers = 19
    # cls.train()
    #
    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v5')
    # cls.layers = 224
    # cls.train()
    #
    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_19_v6')
    # cls.layers = 19
    # cls.train()
    #
    # cls = ClassifierCNN('d169_mura_class_aug_224.h5', 'AUB_WRIST', 'd169_aub_wrist_224_224_v6')
    # cls.layers = 224
    # cls.train()

    # cls = ClassifierCNN(DenseNet169, 'MURA_WRIST', 'd169_mura_hand_test')
    # cls.layers = 224
    # cls.train()

    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_SHOULDER', 'd169_mura_shoulder_224_50_v1')
    # cls.layers = 50
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_FOREARM', 'd169_mura_forearm_224_50_v1')
    # cls.layers = 50
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_SHOULDER', 'd169_mura_shoulder_224_50_v2')
    # cls.layers = 50
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224.h5', 'MURA_FOREARM', 'd169_mura_forearm_224_50_v2')
    # cls.layers = 50
    # cls.train()

    # cls = ClassifierCNN(DenseNet169, 'MURA_WRIST', 'd169_mura_wrist_224_new')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN(DenseNet169, 'MURA_ALL', 'd169_mura_class_224_new')
    # cls.layers = 224
    # cls.train()
    # cls = ClassifierCNN('d169_mura_class_224_new.h5', 'MURA_WRIST', 'd169_mura_wrist_224_224_new')
    # cls.layers = 224
    # cls.train()

    cls = ClassifierCNN(DenseNet169, 'AUB_DIS', 'd169_aub_dis_224_v2')
    cls.layers = 224
    cls.train()

    cls = ClassifierCNN('d169_mura_class_224_new.h5', 'AUB_DIS', 'd169_aub_dis_224_112_v2')
    cls.layers = 112
    cls.train()

# cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_new')   # 0.74317 (ep15)
# cls = ClassifierCNN('d169_mura_class_224_new.h5', 'AUB_WRIST', 'd169_aub_wrist_224_112_new')  # 0.77994 (ep18)
