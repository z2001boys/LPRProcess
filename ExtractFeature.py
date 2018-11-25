
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
from Models import ShuffleNetv2
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


imgObj = LabelImage.DataObj()
imgObj.Rotation    
kerasObj = MyKeras.KerasObj()
kerasObj.label = 'load model'
kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

dataset = "ICDAR03"


SortedClass = LabelMgr.GetAllLabel()

kerasObj.KerasMdl = Inceptionv3.GetMdl((100, 100, 1),len(SortedClass))
kerasObj.KerasMdl.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
kerasObj.LoadWeight("TrainResult/", 'ICDAR03_train_Inceptionv3')

imgObj.LoadList("D:\\DataSet\\"+dataset+"\\trainList.txt",
                            SortedClass=SortedClass)

feature = kerasObj.ExtractFeature(imgObj)
numpy.save("GetNetFeature\\Inceptionv3_ICDAR03_train.npy",feature)
