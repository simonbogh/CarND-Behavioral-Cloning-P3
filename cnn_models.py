'''
Implements different Convolutional Neural Networks.

Currently supported network architectures
- LeNet-5: lenet_model()
- Nvidia self-driving cars: nvidia_model()

'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D, Conv2D
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D


def lenet_model(crop_input=((0, 0), (0, 0))):
    """LeNet-5 architecture.

    - The input data is cropped to remove top and bottom of the images. The top and bottom of the images do not contain much relevant information i.e. trees, sky, and the front of the car.
    - The network architecture consists of two convolutional layers. Max pooling is added after the two convolutional layers.
    - Finally, the neural network has two fully-connected layers of size 120 and 84.
    - The network outputs the logit directly i.e. the steering angle.

    Returns:
        Sequential() -- Returns a Keras Sequential model with a linear stack of layers.
    """
    model = Sequential()
    # Crop image data
    model.add(Cropping2D(cropping=crop_input, input_shape=(160, 320, 3)))
    # Normalize input data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # Conv_1 + max pooling
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    # Conv_2 + max pooling
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    # Flatten before fully-connected layers
    model.add(Flatten())
    # FC_1
    model.add(Dense(120))
    # FC_2
    model.add(Dense(84))
    # Logits: one output (steering angle, float)
    model.add(Dense(1))
    # Return model
    return model


def nvidia_model(crop_input=((0, 0), (0, 0))):
    """ This architecture replicates the Nvidia self-driving cars network architecture from https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

    - The input data is cropped to remove top and bottom of the images. The top and bottom of the images do not contain much relevant information i.e. trees, sky, and the front of the car.
    - The network architecture consists of five convolutional layers. Max pooling is added after the first three convolutional layers.
    - Finally, the neural network has three fully-connected layers of size 100, 50, and 10.
    - The network outputs the logit directly i.e. the steering angle.

    Returns:
        Sequential() -- Returns a Keras Sequential model with a linear stack of layers.
    """
    model = Sequential()
    # Crop image data
    model.add(Cropping2D(cropping=crop_input, input_shape=(160, 320, 3)))
    # Normalize input data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # Conv_1 + max pooling
    model.add(Conv2D(filters=36, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    # Conv_2 + max pooling
    model.add(Conv2D(filters=36, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    # Conv_3 + max pooling
    model.add(Conv2D(filters=48, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    # Conv_4
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    # Conv_5
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    # Flatten before fully-connected layers
    model.add(Flatten())  # Flatten before fully-connected
    # FC_1
    model.add(Dense(100, activation='relu'))
    # FC_2
    model.add(Dense(50, activation='relu'))
    # FC_3
    model.add(Dense(10, activation='relu'))
    # Logits: one output (steering angle, float)
    model.add(Dense(1))  # One output: steering angle (float)
    # Return model
    return model
