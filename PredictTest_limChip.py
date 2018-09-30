import LabelImage
import os
import MyKeras
import tensorflow as tf
import cv2
import ShowImage
import LicensePlateLabel
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


def GetMax(r):
    maxIdx = -1
    maxValue = 0
    for i in range(len(r)):
        if r[i] > maxValue:
            maxIdx = i
            maxValue = r[i]
    return maxIdx, maxValue


'''Load from file'''

testImgObj = LabelImage.DataObj()
trainImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()
kerasObj.LoadAll("TrainResult/", "LimChipping")

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

SortedClass = ["Normal", "Chip"]

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])


result = kerasObj.Evaluate("D:/BinTest/Normal/001.png")


for r in result:
    maxIdx, value = GetMax(r)
    if maxIdx != -1:
        print(SortedClass[maxIdx])

#singleResult = kerasObj.Predict(simpleImg)
#idx = singleResult.argmax()
