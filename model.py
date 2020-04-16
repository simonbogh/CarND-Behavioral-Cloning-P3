"""train_model.py trains a CNN in order to control a self-driving car.

   The idea is to demonstrate Behavioral Cloning, where a user
   has controlled the car and recorded images from a front-facing
   camera, and corresponding steering angles.

   Two different CNN architectures can be loaded and trained from
   cnn_models.py:
   - LeNet
   - Nvidia self-driving cars

   The Nvidia model performed well and succesfully controlled the
   car from 7.998 training samples and 2.000 validation samples.
"""
import csv
import imageio
import numpy as np
import pickle
from sklearn.utils import shuffle
from cnn_models import lenet_model, nvidia_model

import tensorflow as tf
import keras

print("==============================")
print("TensorFlow: ", tf.__version__)
print("Keras: ", keras.__version__)

print("==============================")
print("Loading CSV ..")
lines = []

with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("==============================")
print("Loading images and steering angles ..")
images = []
measurements = []

# Read path information for training images in csv-file
for line in lines:
    # Load original filepath:
    # 0: first column (center image), 1: left image, 2: right image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './training_data/IMG/' + filename

    # Append image
    image = imageio.imread(current_path)
    images.append(image)

    # Append steering angle
    measurement = float(line[3])  # 3: steering angle
    measurements.append(measurement)

    # Augment additional data by flipping image and steering angle
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)

# Save inputs (images) and labels (steering angle) as Numpy arrays
#   model.fit() takes in Numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# Shuffle data
#   Shuffling the data at this point to make sure 'validation_split=' in model.fit()
#   does not only get data that is very similar.
X_train, y_train = shuffle(X_train, y_train)

print("==============================")
print("Initializing NN model ..")
# model = lenet_model(crop_input=((50, 20), (0, 0)))
model = nvidia_model(crop_input=((50, 20), (0, 0)))

print("==============================")
print("Compiling model ..")
model.compile(loss='mse', optimizer='adam')

print("==============================")
print("Training model ..")
history_object = model.fit(X_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, epochs=10)

with open('./trainHistory.p', 'wb') as file_pi:
    pickle.dump(history_object.history, file_pi)

print("==============================")
print("Saving model ..")
model.save('model.h5')

print("==============================")
print("Done")
print("==============================")
