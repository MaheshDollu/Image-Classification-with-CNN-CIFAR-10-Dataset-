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
