# X-Ray Fracture Detection

## How to use

This repository's main code is contained in the `ClassifierCNN` class in `Classifier.py`. All you really need to do is import this class,
set up a proper instance specifying the desired dataset and models, and you're good to go. Some examples are provided here below:

```
# import keras models
from keras.applications.densenet import DenseNet169
from Classifier import ClassifierCNN

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
```
The code above imports the DenseNet169 model and the Classifier class, and then creates a class instance with the desired model to train, a dataset name, and an experiment name. You can also change many of the class variables to suit your need.

The `layers` variable is the number of layers to fine-tune, into the model. The default is set to `19`. For example, at the default value, all model layers will be frozen during training except for the last 224 layers.

Another example, if you want to train the network from scratch (no pretraining), set `layers = 0` and `scratch = True`, as shown above.

The `experiments.py` file provides the same example, and you can modify the file to run your own series of experiments.

You can change many other variables, such as:

* Image Dimensions: `height` and `width` (make sure `height = weight`), default is `224`.
* Batch Size: `batch_size`.
* Loss: `loss`, set to `'default'` for categorical_crossentropy, or to `'focal'` for focal loss.
* Initial Learning Rate: `lrate` with default value of `0.0001`.
* Early Stopping Patience: `es_patience` with default set to `10`.
