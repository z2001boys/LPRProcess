import tensorflow as tf
import LabelImage
import os
import MyKeras
import numpy
import string
import LabelMgr
from Models import MobileNetv1
from Models import MobileNetv2
from Models import Inceptionv3
from Models import ILBPNet
from Models import Xception
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



def Test(NetModel,BenchData):

    # create object
    imgObj = LabelImage.DataObj()
    kerasObj = MyKeras.KerasObj()

    kerasObj.ImageInfo.Size = [100, 100]
    kerasObj.ImageInfo.Channel = 1

    PreProcess = ''
    channerSize = 1
    if NetModel=='ILBPNet':
        PreProcess = "ILBPNet"
        channerSize = 3

    # get labels
    if(BenchData=="CIFAR-100"):
        SortedClass = LabelMgr.GetInt(100)
    else:
        SortedClass = LabelMgr.GetAllLabel()

    # 設定模組
    kerasObj.NewSeq()
    if NetModel != "ILBPNet":
        exec('kerasObj.KerasMdl = '+NetModel+'.GetMdl((100, 100, channerSize),len(SortedClass))' )
    else:
        kerasObj.KerasMdl = ILBPNetv2.GetMdl((100, 100, channerSize),len(SortedClass))#ILBP


    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    kerasObj.KerasMdl.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


    kerasObj.LoadWeight("TrainResult/", BenchData+"_train_"+NetModel)



    imgObj.LoadList("D:\\DataSet\\"+BenchData+"\\testList.txt",
                            SortedClass=SortedClass)



    acc,loss = kerasObj.BenchMark(imgObj,PreProcess=PreProcess,divideSize=5000)

    return acc,loss


