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
kerasObj.LoadAll("TrainResult/", "FirstPlateTest")

kerasObj.ImageInfo.Size = [200, 200]
kerasObj.ImageInfo.Channel = 1

SortedClass = ['Plates']

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])

img = cv2.imread('D:\\License Plate\\108.JPG', 0)
img = cv2.resize(img, (1920, 1080))


detected = kerasObj.FindInImage(img, 0.8, SortedClass)
ShowImage.AddLabel(img, detected)
