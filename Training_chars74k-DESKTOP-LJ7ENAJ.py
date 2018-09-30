import LabelImage
import os
import MyKeras
import tensorflow as tf
import functools


"""keras import
  
""" 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard


# create object
imgObj = LabelImage.DataObj()
testImgObj = LabelImage.DataObj()
kerasObj = MyKeras.KerasObj()
kerasObj.ImageInfo.Size = [100, 100]
kerasObj.ImageInfo.Channel = 1

# Load image
if(os.name == 'nt'):
    imgObj.LoadFromList("DatasetList/TrainingList.txt",
                        TargetSize=(kerasObj.ImageInfo.Size[0],kerasObj.ImageInfo.Size[1]),
                        TrimName="/home/itlab/")
else:
    imgObj.LoadFromList("DatasetList/TrainingList.txt",
                        TargetSize=(kerasObj.ImageInfo.Size[0],
                                    kerasObj.ImageInfo.Size[1]))


# 設定模組
#kerasObj.NewSeq()
kerasObj.LoadModelFromTxt('Models/Inceptionv3.txt')

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
top1_acc = functools.partial(top_k_categorical_accuracy, k=1)
top1_acc.__name__ = 'top1_acc'
top5_acc = functools.partial(top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'

kerasObj.Compile(_optimize='adam',
                 _loss='categorical_crossentropy',
                 _metrics=['accuracy', top1_acc, top5_acc])

# Visualization
log_filepath = '/tmp/chars74k_inceptionv3' 
tb_cb = TensorBoard(log_dir=log_filepath, write_images=1)

cbks = [tb_cb]
kerasObj.Train(imgObj,
               batch_size=32,
               epochs=100,
               verbose=1,
               callbacks=cbks)

# Save the weight
kerasObj.SaveWeight('Weights', 'chars74k_inceptionv3')
'''
testImgObj.LoadFromList("DatasetList/TestList.txt",
                        TrimName="/home/itlab/")
simpleImg = testImgObj.GetImageData(0)
'''
