import tensorflow as tf
import LabelImage
import os
import MyKeras
import numpy
import string
import LabelMgr
"""keras import"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

'''from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))'''


# create object
imgObj = LabelImage.DataObj()
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

# get labels
SortedClass = LabelMgr.GetAllLabel()

# 設定模組
kerasObj.NewSeq()
kerasObj.LoadModelFromTxt('Models\\BasicModel.txt')    

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])

# Load image
if(os.name == 'nt'):
    imgObj.LoadList("DatasetList/TrainingList.txt",
                        SortedClass=SortedClass,
                        TrimName="/home/itlab/")
else:
    imgObj.LoadList("DatasetList/TrainingList.txt",
                        SortedClass=SortedClass)



kerasObj.Train(imgObj,
               SelectMethod='rdn',
               batch_size=128,
               epochs=10,
               rdnSize=1000,
               global_epoche=10,
               verbose=1)

kerasObj.SaveAll("TrainResult/", "chars74k_font")
