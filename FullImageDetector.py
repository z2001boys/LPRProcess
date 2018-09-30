import LabelImage
import os
import MyKeras
import tensorflow as tf
import cv2
import ShowImage
import LicensePlateLabel
import time
import ShowImage
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

kerasObj = MyKeras.KerasObj()
kerasObj.LoadAll("TrainResult/", "chars74k_font")

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

SortedClass = LicensePlateLabel.GetLicenseLabel()

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])

img = cv2.imread('FullImage/1.png', 0)


detected = kerasObj.FindInImage(img, 0.99, SortedClass)
ShowImage.AddLabel(img, detected)
