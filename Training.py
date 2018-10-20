import tensorflow as tf
import LabelImage
import os
import MyKeras
import numpy
import string
import LabelMgr
from Models import MobileNetv2
from Models import Inceptionv3
from Models import DarkNet53
from Models import Inceptionv4

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


def SetTrain(DataSetName,Model,GlobalEpoche,Epoche,rdnSize = 2000,PrePorcess=''):

    # create object
    imgObj = LabelImage.DataObj()
    testImgObj = LabelImage.DataObj()
    kerasObj = MyKeras.KerasObj()

    kerasObj.ImageInfo.Size = [100, 100]
    kerasObj.ImageInfo.Channel = 1

    # get labels
    SortedClass = LabelMgr.GetAllLabel()

    channerSize = 1
    if PrePorcess=="ILBPNet":
        channerSize = 16

    # 設定模組
    kerasObj.NewSeq()
    if PrePorcess != "ILBPNet":
        exec('kerasObj.KerasMdl = '+Model+'.GetMdl((100, 100, channerSize),len(SortedClass))' )
    else:
        kerasObj.KerasMdl = MobileNetv2.GetMdl((100, 100, channerSize),len(SortedClass))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    kerasObj.Compile(_optimize='rmsprop',
                    _loss=sgd,  # 'categorical_crossentropy',
                    _metrics=['accuracy'])

    # Load image
    if(os.name == 'nt'):
        imgObj.LoadList("D:\\DataSet\\"+DataSetName+"\\trainList.txt",
                            SortedClass=SortedClass,
                            TrimName="/home/itlab/")
    else:
        imgObj.LoadList("DatasetList/TrainingList.txt",
                            SortedClass=SortedClass)



    kerasObj.Train(imgObj,
                SelectMethod='rdn',
                batch_size=128,
                epochs=Epoche,
                rdnSize=rdnSize,
                global_epoche=GlobalEpoche,
                PreProcess=PrePorcess,
                verbose=1)

    kerasObj.SaveWeight("TrainResult/", "MNIST_train_DarkNet53")

