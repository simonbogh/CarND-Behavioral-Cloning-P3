import csv
import imageio
import numpy as np

print("==============================")
print("Loading CSV")
lines = []
with open('./training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("==============================")
print("Loading images and steering angles")
images = []
measurements = []
for line in lines:
    source_path = line[0]  # 0: first column (center image), 1: left image, 2: right image
    filename = source_path.split('/')[-1]
    current_path = './training_data/IMG/' + filename

    # Append image
    image = imageio.imread(current_path)
    images.append(image)
    # Append steering angle
    measurement = float(line[3])  # 3: steering angle
    measurements.append(measurement)

    # Flip image and steering angle
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)


X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from cnn_models import custom_model, nvidia_model

print("==============================")
print("Initializing NN model ..")
# model = Sequential()
# model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape=(160, 320, 3)))  # , input_shape=(160,320,3)
# model.add(Lambda(lambda x: (x / 255.0) - 0.5))  # Normalize input data, , input_shape=(160, 320, 3)
# model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())  # Flatten before fully-connected
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))  # One output: steering angle (float)

model = nvidia_model()


print("Compiling model ..")
model.compile(loss="mse", optimizer='adam')
print("Training model ..")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')

print("==============================")
print("Done")
print("==============================")
