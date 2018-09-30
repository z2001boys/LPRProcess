import LabelImage
import os
import MyKeras
import tensorflow as tf
import numpy
import string
import LicensePlateLabel
"""keras import"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3


# create object
imgObj = LabelImage.DataObj()
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

# get labels
SortedClass = ["Normal", "Chip"]

Model = "VGG"

# 設定模組
if Model == "own":
    kerasObj.NewSeq()
    kerasObj.LoadModelFromTxt('Models\\BasicModel_LimChip.txt')

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    kerasObj.Compile(_optimize='adam',
                    _loss='categorical_crossentropy',
                    _metrics=['accuracy'])

if Model =="VGG":
    model = VGG16(weights='imagenet',include_top=False)
    


labels, imgPaths = imgObj.CreateAccessCache(
    'D:/BinTest/', SortedClass)
imgObj.GenImage(
    imgPaths, labels, SortedClass, kerasObj.ImageInfo, eachSize=(1, 1))
imgObj.LoadFromList(
    'D:/tempLabel.txt', (100, 100), SortedClass)

kerasObj.Train(imgObj, batch_size=100, epochs=5)

# kerasObj.TrainByGener('D:/LicensePlateDataSet',
#                      SortedClass, epochs=10, GlobalEpochs=1)


kerasObj.SaveAll("TrainResult/", "LimChipping")
