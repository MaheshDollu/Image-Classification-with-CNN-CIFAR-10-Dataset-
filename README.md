#  Image Classification with CNN (CIFAR-10 Dataset)

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

# Building the CNN model

model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    
    Dropout(0.5),
    
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

Conv2D: This is a 2D convolutional layer. It learns filters (or kernels) that detect specific features (such as edges or textures) in the images.

The first Conv2D layer has 32 filters of size 3x3, using the ReLU activation function. It expects input images of shape (32, 32, 3) (32x32 images with 3 color channels).

The second Conv2D layer has 64 filters of size 3x3.

The third Conv2D layer has 128 filters of size 3x3.

MaxPooling2D: This is a pooling layer that reduces the spatial dimensions (height and width) of the feature maps after the convolutional layers. It helps in reducing the computational load and in making the model more robust.

Each MaxPooling2D layer uses a 2x2 window, which reduces the size of the image by half in both dimensions.

Flatten: This layer flattens the 3D output of the previous convolutional and pooling layers into a 1D vector. This is necessary before feeding the output into the fully connected Dense layers.

Dense: These are fully connected layers.

The first Dense layer has 128 neurons and uses the ReLU activation function.

The final Dense layer has 10 neurons (corresponding to the 10 classes in CIFAR-10), with a softmax activation function to output a probability distribution over the 10 classes.

Dropout: This layer randomly drops a percentage (50% in this case) of the neurons during training to prevent overfitting. It helps to generalize the model better.

# Compiling the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

optimizer='adam': Adam is an adaptive learning rate optimization algorithm that computes adaptive learning rates for each parameter and helps in faster convergence.

loss='categorical_crossentropy': This is the loss function used for multi-class classification problems. It measures how well the model's predictions match the actual labels.

metrics=['accuracy']: This specifies that the model should report accuracy during training and evaluation.

# Training the model

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

x_train, y_train: The training data and labels.

epochs=10: The model will train for 10 epochs (iterations through the entire training dataset).

batch_size=64: The model will process 64 samples at a time before updating the weights.

validation_data=(x_test, y_test): The model will also validate its performance on the test data after each epoch.

# Evaluating the model

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_acc}")

After training, the model is evaluated on the test data using the evaluate method, which computes the loss and accuracy of the model on the test set.

test_loss: The loss on the test data.

test_acc: The accuracy on the test data.
