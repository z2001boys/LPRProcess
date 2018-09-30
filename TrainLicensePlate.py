import LabelImage
import os
import MyKeras
import tensorflow as tf
import numpy
import string
import LicensePlateLabel
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3


# create object
imgObj = LabelImage.DataObj()
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [200, 200]
kerasObj.ImageInfo.Channel = 1

# get labels
SortedClass = ["Plates", "Non_Plates"]


# 設定模組
kerasObj.NewSeq()
kerasObj.LoadModelFromTxt('Models\\PlateModel_My.txt')

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])

labels, imgPaths = imgObj.CreateAccessCache(
    'D:/LicensePlateDataSet', SortedClass)
imgObj.GenImage(
    imgPaths, labels, SortedClass, kerasObj.ImageInfo, eachSize=(10000, 10000))
imgObj.LoadFromList(
    'D:/tempLabel.txt', (200, 200), ["Plates", "Non_Plates"])

kerasObj.Train(imgObj, batch_size=100, epochs=20)

# kerasObj.TrainByGener('D:/LicensePlateDataSet',
#                      SortedClass, epochs=10, GlobalEpochs=1)


kerasObj.SaveAll("TrainResult/", "FirstPlateTest")
