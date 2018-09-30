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
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


# create object
imgObj = LabelImage.DataObj()
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

# get labels
SortedClass = LicensePlateLabel.GetLicenseLabel()

# Load image
if(os.name == 'nt'):
    imgObj.LoadFromList("DatasetList/TrainingList.txt",
                        SortedClass=SortedClass,
                        TrimName="/home/itlab/")
else:
    imgObj.LoadFromList("DatasetList/TrainingList.txt",
                        SortedClass=SortedClass)


# 設定模組
kerasObj.NewSeq()
kerasObj.LoadModelFromTxt('Models\\BasicModel_LimChip.txt')

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])


kerasObj.Train(imgObj,
               batch_size=128,
               epochs=10,
               verbose=1)

kerasObj.SaveAll("TrainResult/", "chars74k_font")
