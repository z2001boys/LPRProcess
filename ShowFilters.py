import tensorflow as tf
import LabelImage
import os
import MyKeras
import numpy
import string
import LabelMgr
import math
from Models import MobileNetv1
from Models import MobileNetv2
from Models import Inceptionv3
from Models import DarkNet53
from Models import Inceptionv4
from Models import ILBPNet
from Models import Xception
from Models import BasicModel
from Models import DenseNet169
from Models import DarkNet53
from Models import ResNet50
from Models import ILBPNetv2
from Models import ILBPNetv3
from Models import ShuffleNetv2
from Models import SqueezeNet
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD




# create object
imgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()
kerasObj.label = "test"
kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1
# get labels
if(DataSetName=="CIFAR-100"):
    SortedClass = LabelMgr.GetInt(100)
else:
    SortedClass = LabelMgr.GetAllLabel()
channerSize = 1
if "ILBPNet" in Model:
    PrePorcess = Model
    channerSize = 3
else:
    PrePorcess = Model
# 設定模組
kerasObj.NewSeq()
if KerasLoadModel=='':
    KerasLoadModel = Model
if "ILBPNet" not in KerasLoadModel:
    exec('kerasObj.KerasMdl = '+KerasLoadModel+'.GetMdl((100, 100, channerSize),len(SortedClass))' )
elif "+" in KerasLoadModel:
    kerasObj.KerasMdl = ILBPNetv3.GetMdl((10000+3240+2250,1),len(SortedClass))#ILBP
else:
    kerasObj.KerasMdl = ILBPNetv2.GetMdl((100, 100, channerSize),len(SortedClass))#ILBP

kerasObj.KerasMdl.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])