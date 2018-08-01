import cv2 as cv 
import numpy as np

import tensorflow as tf 
from tensorflow import keras 

import data_processing

from sklearn.model_selection import train_test_split

# 0~9 and A~Z (except I and O)
num_classes = 34
img_rows, img_cols = 64, 64
batch_size = 50
epochs = 10

# Load the data 
print("Loading data ...")

# Load training data and label
data_processing.extract_label_file('/home/itlab/English/Fnt', '/home/itlab/output.txt')
x, y = data_processing.load_from_label_text('/home/itlab/output.txt')

# Load test data and label
data_processing.extract_label_file('/home/itlab/test_img', '/home/itlab/test_output.txt')
x_test, y_test = data_processing.load_from_label_text('/home/itlab/test_output.txt')
print('Data has been loaded.')

# Split dataset to training set and validation set
x_train, x_validation, y_train, y_validation = train_test_split(x, y, random_state=2, train_size=0.8)

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_validation = x_validation.reshape(x_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')
x_train /= 255
x_test /= 255
x_validation /= 255

# Convert class vector to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)

# Convolutional Network architecture
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model
adam = keras.optimizers.Adam(lr=0.001, decay=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model 
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

# Evaluate accuracy 
score = model.evaluate(x_validation, y_validation, verbose=0)

print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# Make predictions
tp = 0

predictions = model.predict(x_test)

for test_count in range(0, x_test.shape[0]):
    prediction = predictions[test_count]
    y_pred = np.argmax(prediction)
    
    if (y_pred == y_test[test_count]):
        tp += 1
    
tpr = tp / x_test.shape[0]
print("Test accuracy: {:2.4%}".format(tpr))
    
# Save the model 
model.save('/tmp/chars74k_model.h5')
