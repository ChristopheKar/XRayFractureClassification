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

# cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_new')   # 0.74317 (ep15)
# cls = ClassifierCNN('d169_mura_class_224_new.h5', 'AUB_WRIST', 'd169_aub_wrist_224_112_new')  # 0.77994 (ep18)
# cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_scratch_new') # 0.78309 (ep26)   (0.74947 at ep8)
# cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_scratch_new') 0.69223 (ep43)
# cls = ClassifierCNN(DenseNet169, 'MURA_ALL', 'd169_mura_class_scratch.h5') 0.84384 (ep57)

cls = ClassifierCNN('d169_mura_class_224_new.h5', 'AUB_DISP', 'd169_aub_dis_224_112_relabeled')
cls.layers = 112
cls.train()

cls = ClassifierCNN('d169_aub_wrist_224_relabeled.h5', 'AUB_DISP2', 'd169_aub_dis_224_112f')
cls.layers = 112
cls.train()

# cls = ClassifierCNN(DenseNet169, 'AUB_DIS', 'd169_aub_dis_224_v1')   # 0.66682 (ep29, softmax)
# cls = ClassifierCNN('d169_mura_class_224_new.h5', 'AUB_DIS', 'd169_aub_dis_224_112_v1')   # 0.68113 (ep27, softmax)
# ^ same with focal loss: 0.66636 (ep34), 0.67974 (ep33)
# ^ same with focal loss and balanced classes (0.66636 ep 34)
