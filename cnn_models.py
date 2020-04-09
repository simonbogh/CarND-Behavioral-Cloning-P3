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
