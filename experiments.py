# import keras models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from Classifier import ClassifierCNN

if __name__ == '__main__':

    cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v1')
    cls.layers = 224
    cls.train()
    cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v2')
    cls.layers = 224
    cls.train()
    cls = ClassifierCNN(DenseNet169, 'AUB_WRIST', 'd169_aub_wrist_224_v3')
    cls.layers = 0
    cls.scratch = True
    cls.train()
