import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array

models_root = '/home/ubuntu/wrist/classification/models'

conv = [47, 135, 363, 591]
model_paths = ['d169_aub_wrist_224_relabeled.h5',
               'd169_aub_wrist_224_112_relabeled.h5',
               'd169_aub_wrist_scratch_relabeled.h5',
               'd169_aub_wrist_0_224_relabeled.h5']

img_path = '/home/ubuntu/wrist/datasets/aub_fracture/train/fracture/755914_new.png'
img = load_img(img_path, target_size=(224,224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img*1./255

for m in model_paths[-1]:
    print('Loading model {:s}...'.format(m))
    model = load_model(os.path.join(models_root, m))
    layer_outputs = [model.layers[0].layers[c].output for c in conv]
    layer_names = [model.layers[0].layers[c].name for c in conv]
    activation_model = Model(inputs=model.layers[0].inputs, outputs=layer_outputs
    print('Running activations...')
    activations = activation_model.predict(img)
    fig_names = []
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        fig_name = 'feature_map_' + layer_name + '.png'
        fig_names.append(fig_name)
        plt.savefig(fig_name)

    im = Image.fromarray(np.vstack([np.array(Image.open(path)) for path in fig_names]))
    for path in fig_names:
        os.remove(path)
    im.save('feature_maps_' + m + '.png')
    del activation_model
    del model
