


import tensorflow as tf
import LabelImage
import os
import MyKeras
import numpy
import string
import LabelMgr
from Models import MobileNetv2
from Models import Inceptionv3
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


# create object
imgObj = LabelImage.DataObj()
imgObj.Rotation = 5
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()

kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

# get labels
SortedClass = LabelMgr.GetAllLabel()

# 設定模組
kerasObj.NewSeq()
kerasObj.KerasMdl = MobileNetv2.GetMdl((100, 100, 16),len(SortedClass)) 


sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
kerasObj.Compile(_optimize='rmsprop',
                 _loss=sgd,  # 'categorical_crossentropy',
                 _metrics=['accuracy'])


kerasObj.LoadWeight("TrainResult/", "IIIT5K_train_ILBPNet")


imgObj.LoadList("D:\\DataSet\\IIIT5K\\testList.txt",
                        SortedClass=SortedClass)


correct = []
loss = []
for r in range(-90,91,15):
    imgObj.Rotation = r
    c,l = kerasObj.BenchMark(imgObj,PreProcess="ILBPNet",divideSize=5000)
    correct.append(c)
    loss.append(l)


with open('D:\\correct.txt', 'w') as f:
    for item in correct:
        f.write("%s\n" % item)

with open('D:\\loss.txt', 'w') as f:
    for item in loss:
        f.write("%s\n" % item)