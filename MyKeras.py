import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import LabelImage
import ImageInfoClass as IF


class KerasObj:
    def __init__(self, _ImageSize=[100,100], _ImageChannel=1, _ImageFlatten=False):
        self.KerasMdl = []
        self.Layers = []
        #define input data
        self.ImageInfo = IF.ImageInfoClass()
        self.ImageInfo.Size = _ImageSize    
        self.ImageInfo.Channel = _ImageChannel
        self.ImageInfo.NeedFlatten = _ImageFlatten
        

    def NewSeq(self):
        self.KerasMdl = Sequential()
        self.Layers = []

    def AddLayer(self, fn):
        if self.KerasMdl != []:
            self.KerasMdl.add(fn)
            self.Layers.append(fn._name)

    # load models

    def LoadMdl(self, _loadPath, _modelName):
        _loadPath = LabelImage.PathCheck(_loadPath)
        with open(_loadPath+_modelName, 'r') as binStream:
            self.model = tf.keras.models.model_from_json(binStream.read())

    # load weight(with h5py, please install h5py with pip)
    def LoadWeight(self, _loadPath, _weightName):
        _loadPath = LabelImage.PathCheck(_loadPath)
        if self.mldInit == True:
            self.model.load_weights(_loadPath)

    def Train(self, imgObj, batch_size, epochs, verbose):
        print("Prepare data...")
        imageData = imgObj.GetImageData(self.ImageInfo,Dim = 4)
        label, _ = imgObj.ToIntLable(ArrayExtend=True)


        self.KerasMdl.fit(imageData, label, batch_size, epochs, verbose)



    def Compile(self, _optimize, _loss,_metrics):
        self.KerasMdl.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])