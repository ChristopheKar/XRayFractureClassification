from keras.models import load_model
from keras.preprocessing import image
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(img_path):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    return img_tensor

if __name__ == '__main__':

  model = load_model("models/vgg_mura_autoenc.h5")
  img_path = '/home/ubuntu/wrist/datasets/MURA_classification/valid/wrist/image1763.png'
  new_image = load_image(img_path)
  pred = model.predict(new_image)
  imsave('test.png', pred[0])
