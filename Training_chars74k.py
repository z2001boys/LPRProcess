import LabelImage
import os
import MyKeras
import tensorflow as tf
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


# create object
imgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [100,100]
kerasObj.ImageInfo.Channel = 1


# Load image
if(os.name == 'nt'):
    imgObj.LoadFromList("TrainingList.txt", TrimName="/home/itlab/")

else:
    imgObj.LoadFromList("TrainingList.txt")


# 設定模組
kerasObj.NewSeq()
kerasObj.AddLayer(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
kerasObj.AddLayer(Conv2D(32, (3, 3), activation='relu'))
kerasObj.AddLayer(MaxPooling2D(pool_size=(2, 2)))
kerasObj.AddLayer(Dropout(0.25))

kerasObj.AddLayer(Conv2D(64, (3, 3), activation='relu'))
kerasObj.AddLayer(Conv2D(64, (3, 3), activation='relu'))
kerasObj.AddLayer(MaxPooling2D(pool_size=(2, 2)))
kerasObj.AddLayer(Dropout(0.25))

kerasObj.AddLayer(Flatten())
kerasObj.AddLayer(Dense(1024, activation='relu'))
kerasObj.AddLayer(Dropout(0.5))
kerasObj.AddLayer(Dense(34, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile( _optimize=sgd,
              _loss='binary_crossentropy',
              _metrics=['accuracy'] )


kerasObj.Train(imgObj,
               batch_size=300,
               epochs=110,
               verbose=1)
