import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define some things
num_classes = 10
input_shape = (28, 28, 1)

# Load MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and scale data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu", data_format="channels_last"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train model and evaluate performance
model.fit(x_train, y_train, batch_size=128, epochs=4)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model
model.save(r"/home/pi/CPE4903/digit_recog.h5")
