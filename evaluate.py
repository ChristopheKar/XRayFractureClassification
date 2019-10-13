import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# ================================== INPUTS ====================================
model_name = 'd169_aub_dis_224_112_v1'
VAL_DIR = '/home/ubuntu/wrist/datasets/MURA_wrist/valid'
# ==============================================================================

if not model_name.endswith('.h5'):
    model_name = model_name + '.h5'

model_path = os.path.join('/home/ubuntu/wrist/classification/models', model_name)
m = load_model(model_path)

WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 1

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size = (HEIGHT, WIDTH),
    batch_size = BATCH_SIZE,
    shuffle = False,
    class_mode = 'categorical')


validation_generator.reset()
classes = validation_generator.classes[validation_generator.index_array][0]
nb_samples = len(classes)

validation_generator.reset()
Y_pred = m.predict_generator(validation_generator, steps=nb_samples)
pred_prob = np.array([a[0] for a in Y_pred])
pred_classes = pred_prob.round().astype('int32')


# accuracy: (tp + tn) / (p + n)
val_acc = accuracy_score(classes, pred_classes)
# precision tp / (tp + fp)
precision = precision_score(classes, pred_classes)
# recall: tp / (tp + fn)
recall = recall_score(classes, pred_classes)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(classes, pred_classes)
# kappa
kappa = cohen_kappa_score(classes, pred_classes)
# ROC AUC
auc = roc_auc_score(classes, pred_prob)
# confusion matrix
matrix = confusion_matrix(classes, pred_classes)

print('Accuracy: %f' % val_acc)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1)
print('Cohens kappa: %f' % kappa)
print('ROC AUC: %f' % auc)
print(matrix)

max_acc = 0
max_thresh = 0
pred_classes = np.array([])
for thresh in np.arange(0.1, 1, 0.01):
    for a in pred_prob:
        if a > thresh:
            pred_classes = np.append(pred_classes, 1)
        else:
            pred_classes = np.append(pred_classes, 0)
    val_acc = accuracy_score(classes, pred_classes)
    if val_acc > max_acc:
        max_acc = val_acc
        max_thresh = thresh

print('Max Acccuracy is {:f} for threshold at {:f}'.format(val_acc, thresh))
