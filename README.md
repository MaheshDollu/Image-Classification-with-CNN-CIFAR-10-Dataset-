# Sentiment-Analysis-Using-RNN-IMDB-Dataset-

# Importing necessary libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

TensorFlow: This is the deep learning framework used for building, training, and evaluating the model.

Sequential: A linear stack of layers used to define the model.

Conv2D, MaxPooling2D, Flatten, Dense, Dropout: Layers used to build the CNN architecture. These are the core building blocks of the network.

cifar10: The dataset of images used for training and testing.

to_categorical: Used to convert the labels (target classes) into one-hot encoded vectors, as required for classification tasks.

# Loading and preprocessing the CIFAR-10 datase

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode

cifar10.load_data(): Loads the CIFAR-10 dataset and splits it into training and test sets.


x_train and x_test are the image data (each image is a 32x32 RGB image).

y_train and y_test are the corresponding labels (classifications of images, ranging from 0 to 9).

Normalization: The images' pixel values range from 0 to 255. Dividing them by 255 scales the values to the range [0, 1], which helps in faster convergence during training.


One-hot encoding: The labels (which are integers from 0 to 9) are converted into one-hot encoded vectors, so each label will be represented as a 10-element vector where only the correct class is 1 and all others are 0 (e.g., class 0 would be [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).
