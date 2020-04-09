'''
Different models
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def custom_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape=(160, 320, 3)))  # , input_shape=(160,320,3)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))  # Normalize input data, , input_shape=(160, 320, 3)
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())  # Flatten before fully-connected
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))  # One output: steering angle (float)
    return model


def nvidia_model():
    """ This architecture replicates the Nvidia self-driving network architecture from https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

    - The input data is cropped to remove top and bottom of the images. The top and bottom of the images do not contain much relevant information i.e. trees, sky, and the front of the car.
    - The network architecture consists of five convolutional layers. Max pooling is added after the first three convolutional layers.
    - Finally, the neural network has three fully-connected layers of size 100, 50, and 10.
    - The network outputs the logit directly i.e. the steering angle.

    Returns:
        Sequential() -- Returns a Keras Sequential model with a linear stack of layers.
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape=(160, 320, 3)))  # , input_shape=(160,320,3)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))  # Normalize input data, , input_shape=(160, 320, 3)
    model.add(Conv2D(filters=36, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())  # Flatten before fully-connected
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # One output: steering angle (float)
    return model
