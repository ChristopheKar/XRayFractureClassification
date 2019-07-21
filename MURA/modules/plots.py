#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:15:06 2018

@author: michal
"""

# =============================================================================
#                                    Plots
# =============================================================================

from keras.preprocessing import image
import matplotlib.pyplot as plt

def plot_sample_img():

    plt.figure(1,figsize=(25,12))
    plt.suptitle("XR_HAND",fontsize=30)

    plt.subplot(1,3,1)
    img_path = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image1.png')
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title("image1")
    plt.xlabel("Normal",fontsize=15)

    plt.subplot(1,3,2)
    img_path1 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image2.png')
    img1 = image.load_img(img_path1)
    plt.imshow(img1)
    plt.title("image2")
    plt.xlabel("Normal",fontsize=15)


    plt.subplot(1,3,3)
    img_path2 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image4.png')
    img2 = image.load_img(img_path2)
    plt.imshow(img2)
    plt.title("image3")
    plt.xlabel("Normal",fontsize=15)

    plt.savefig("Sample.png")



def plot_training(history,save_fig,name,val = True):

    """
    save_fig - figure path
    name - figure name
     """

    acc = history.history["acc"]
    loss = history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.figure(figsize=(20,10))

    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    if save_fig:
         plt.savefig(save_fig + str("/") + str(name))

'''
def plot_predictions(pic_number,X_test,y_test,y_pred):

    plt.figure()
    plt.suptitle("Cancer detection system",fontsize = 20)
    plt.subplot(1,2,1)
    plt.imshow(X_test[pic_number])
    print(int(y_test[pic_number]))

    if y_test[pic_number] == 1:
        plt.title("Diagnosis : " + "positive")
    else :
        plt.title("Diagnosis : " + "negative")

    plt.subplot(1,2,2)
    plt.imshow(X_test[pic_number])
    print(int(y_pred[pic_number]))

    if y_pred[pic_number] == 1:
        plt.title("Prediction : " + "positive")
    else :
        plt.title("Prediction : " + "negative")
'''
