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


def SetTrain(DataSetName,Model,
    batchSize = 128,
    GlobalEpoche=10,Epoche=10,rdnSize = 2000,
    skLearn = False):

    PrePorcess = ''

    # create object
    imgObj = LabelImage.DataObj()
    imgObj.Rotation
    testImgObj = LabelImage.DataObj()
    kerasObj = MyKeras.KerasObj()

    kerasObj.ImageInfo.Size = [100, 100]
    kerasObj.ImageInfo.Channel = 1

    # get labels
    if(DataSetName=="CIFAR-100"):
        SortedClass = LabelMgr.GetInt(100)
    else:
        SortedClass = LabelMgr.GetAllLabel()


    channerSize = 1
    if Model=="ILBPNet":
        PrePorcess = "ILBPNet"
        channerSize = 3

    # 設定模組
    kerasObj.NewSeq()
    if Model != "ILBPNet":
        exec('kerasObj.KerasMdl = '+Model+'.GetMdl((100, 100, channerSize),len(SortedClass))' )
    else:
        kerasObj.KerasMdl = ILBPNetv2.GetMdl((100, 100, channerSize),len(SortedClass))#ILBP

    kerasObj.KerasMdl.summary()

    
    kerasObj.KerasMdl.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Load image
    if(os.name == 'nt'):
        imgObj.LoadList("D:\\DataSet\\"+DataSetName+"\\trainList.txt",
                            SortedClass=SortedClass,
                            TrimName="/home/itlab/")
    else:
        imgObj.LoadList("DatasetList/TrainingList.txt",
                            SortedClass=SortedClass)
    
    kerasObj.ValidateSplit = 0.1

    if GlobalEpoche == -1:
        imgSize = len(imgObj.ImgList)
        GlobalEpoche = math.ceil( imgSize/rdnSize )

    if skLearn==True:
        kerasObj.TrainBySvm(imgObj,
                    SelectMethod='rdn',
                    batch_size=128,
                    epochs=Epoche,
                    rdnSize=rdnSize,
                    global_epoche=GlobalEpoche,
                    PreProcess=PrePorcess,
                    verbose=1,
                    savePath=Model+'_'+DataSetName)
    else:
        kerasObj.Train(imgObj,
                    SelectMethod='rdn',
                    batch_size=128,
                    epochs=Epoche,
                    rdnSize=rdnSize,
                    global_epoche=GlobalEpoche,
                    PreProcess=PrePorcess,
                    verbose=1,
                    savePath=Model+'_'+DataSetName)

    kerasObj.SaveWeight("TrainResult/", DataSetName+"_train_"+Model)

