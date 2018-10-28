


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
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


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
        channerSize = 16

    # get labels
    SortedClass = LabelMgr.GetAllLabel()

    # 設定模組
    kerasObj.NewSeq()
    if NetModel != "ILBPNet":
        exec('kerasObj.KerasMdl = '+NetModel+'.GetMdl((100, 100, channerSize),len(SortedClass))' )
    else:
        kerasObj.KerasMdl = ILBPNet.GetMdl((100, 100, channerSize),len(SortedClass))#ILBP


    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    kerasObj.Compile(_optimize='rmsprop',
                    _loss=sgd,  # 'categorical_crossentropy',
                    _metrics=['accuracy'])


    kerasObj.LoadWeight("TrainResult/", BenchData+"_train_"+NetModel)


    imgObj.LoadList("D:\\DataSet\\"+BenchData+"\\testList.txt",
                            SortedClass=SortedClass)



    acc,loss = kerasObj.BenchMark(imgObj,PreProcess=PreProcess,divideSize=5000)

    return acc,loss


