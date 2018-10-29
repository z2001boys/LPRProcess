import tensorflow as tf
from tensorflow.keras import backend as K
import LabelImage
import ImageInfoClass as IF
import CreateImagePatches as CIP
import numpy as np
import cv2
import CommonMath
import CommonStruct
import math
import ILBPLayer
import gc
import os
from matplotlib import pyplot as plt
'''keras import'''
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Input,AveragePooling2D,concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from ILBPLayer import MyLayer
import pickle

class KerasObj:
    def __init__(self, _ImageSize=[100, 100], _ImageChannel=1, _ImageFlatten=False):
        self.KerasMdl = []
        self.Layers = []
        self.labelDict = []
        self.LayerNum = 0
        # define input data
        self.ImageInfo = IF.ImageInfoClass()
        self.ImageInfo.Size = _ImageSize
        self.ImageInfo.Channel = _ImageChannel
        self.ImageInfo.NeedFlatten = _ImageFlatten
        self.ValidateSplit = 0.0

    def NewSeq(self):
        self.KerasMdl = Sequential()
        self.Layers = []

    def AddLayer(self, fn):
        if self.KerasMdl != []:
            self.KerasMdl.add(fn)
            self.Layers.append(fn._name)

    # load models
    def LoadAll(self, _loadPath, _name):
        _loadPath = LabelImage.PathCheck(_loadPath)
        self.KerasMdl = tf.keras.models.load_model(_loadPath+_name+"_mdl.h5")
        self.KerasMdl.load_weights(_loadPath+_name+'_weight.h5')

    def LoadMdl(self, _loadPath, _modelName):
        _loadPath = LabelImage.PathCheck(_loadPath)
        self.KerasMdl = tf.keras.models.load_model(
            _loadPath+_modelName+"_mdl.h5")

    # load weight(with h5py, please install h5py with pip)
    def LoadWeight(self, _loadPath, _weightName):
        _loadPath = LabelImage.PathCheck(_loadPath)
        self.KerasMdl.load_weights(_loadPath+_weightName+'_weight.h5')

    def SaveAll(self, _savePath, _name):
        _savePath = LabelImage.PathCheck(_savePath)
        self.KerasMdl.save(_savePath+_name+"_mdl.h5")
        self.KerasMdl.save_weights(_savePath+_name+"_weight.h5")

    def SaveMdl(self, _savePath, _modelName):
        _savePath = LabelImage.PathCheck(_savePath)
        self.KerasMdl.save(_savePath+_modelName+"_mdl.h5")

    # load weight(with h5py, please install h5py with pip)
    def SaveWeight(self, _savePath, _weightName):
        _savePath = LabelImage.PathCheck(_savePath)
        self.KerasMdl.save_weights(_savePath+_weightName+"_weight.h5")

    def BenchMark(self,imgObj,PreProcess = '',divideSize = 1000):

        listSize = imgObj.GetListSize()
        DoTimes = math.ceil(listSize/divideSize)
        correct = 0
        loss = 0
        
        for i in range(DoTimes):
            pickIdx = list(range(i*divideSize,min((i+1)*divideSize,listSize)))
            img,label = imgObj.RadomLoad(self.ImageInfo,PickSize=len(pickIdx), Dim=4 , PreProcess = PreProcess,randIdx = pickIdx,kerasLabel=True)
            p = self.KerasMdl.evaluate(img,label)
            print('current:' ,p)
            correct = p[1]+correct
            loss = loss+p[0]
        return correct/DoTimes,loss/DoTimes

    def Train(self, imgObj, SelectMethod='all',rdnSize = -1 ,
        global_epoche = 1,
        PreProcess = '',
        batch_size=32, epochs=1, verbose=0,valitationSplit = 0.0,
        savePath = ''):

        #若RDN==-1則使用全部的影像
        if rdnSize == -1:
            rdnSize = imgObj.GetListSize()

        for i in range(global_epoche):
            print("Start Training")
            print("==Total layer : ", self)
            print("==Global Epoche : ", i)
            print("Prepare data...-")
            imageData,label = imgObj.RadomLoad(self.ImageInfo,PickSize=rdnSize, Dim=4 , PreProcess = PreProcess)
                
        
            singleResult = self.KerasMdl.fit(imageData,label,
                batch_size = batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_split=self.ValidateSplit )

            if savePath != '':
                with open('TrainHistory\\'+savePath+'_', 'w') as file_pi:
                    pickle.dump(singleResult.history, file_pi)
            
            gc.collect()#clear previous image data


    def TrainByGener(self, trainPath,
                     IncludeSet=[],
                     TrimName = '',
                     LoadSize = 50, #每一次讀入的子集影像數量
                     GenerSize = 3000,
                     GlobalEpochs=10,
                     GlobalBatchSize=3000,
                     batch_size=-1, epochs=-1, verbose=-1):
        if IncludeSet == []:
            raise Exception('Data Set cant be empty')

        imgObj = LabelImage.DataObj()
        if os.path.isfile( trainPath )==False:
            labels, imgPaths = imgObj.CreateAccessCache(trainPath, IncludeSet)
        else:
            labels, imgPaths = imgObj.CreateAccessCacheByFile(trainPath, IncludeSet,TrimName=TrimName)

        baseEpoch = GenerSize

        #prepare each learning size
        eachSize = []
        for i in range(0,len(IncludeSet)):
            eachSize.append(LoadSize)

        for i in range(GlobalEpochs):
            imgObj.GenImage(
                imgPaths, labels, IncludeSet, self.ImageInfo, eachSize=eachSize)
            imgObj.LoadFromList(
                'D:/tempLabel.txt', (self.ImageInfo.Size[0], self.ImageInfo.Size[1]), IncludeSet)

            # get learning data
            x_train = imgObj.GetImageData(self.ImageInfo, Dim=4)
            y_train, _ = imgObj.ToIntLable(FinalTarget="ArrayExtend")


            print("Globle Epoche run:",i)

            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            datagen.fit(x_train)

            self.KerasMdl.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                        steps_per_epoch=2000, epochs=epochs)

            # prepare for next round
            baseEpoch = baseEpoch+5

            gc.collect()
            continue  # for next round

        return 1

    def Compile(self, _optimize, _loss, _metrics):
        self.KerasMdl.compile(optimizer='rmsprop',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

    def FindInImage(self, img, score, labels):
        imgPatch, BoxSize, boxR, boxC = CIP.CreatePatch(
            img, Stride=(80, 80), TargetSize=(200, 200))
        box = []
        MatchTable = self.KerasMdl.predict(imgPatch)
        for i in range(MatchTable.shape[1]):
            scoreMap = np.reshape(MatchTable[:, i], (BoxSize[0], BoxSize[1]))
            loc = np.where(scoreMap > score)

            for j in range(len(loc[0])):
                targetRow = loc[0][j]
                targetCol = loc[1][j]
                # local maxima means it is greater than neighbor
                if CommonMath.LocalMaxima(scoreMap, targetRow, targetCol):
                    idx = targetRow*BoxSize[1]+targetCol
                    CommonMath.CalcAccuracyPoint(
                        scoreMap, targetRow, targetCol)

                    # add a label object
                    newLabel = CommonStruct.ObjectLabel()

                    newLabel.Label = labels[i]
                    newLabel.Score = scoreMap[targetRow, targetCol]
                    newLabel.Box.r1 = boxR[idx]
                    newLabel.Box.c1 = boxC[idx]
                    newLabel.Box.r2 = boxR[idx]+100
                    newLabel.Box.c2 = boxC[idx]+100

                    box.append(newLabel)

                    '''cv2.rectangle(
                        img, (boxC[idx], boxR[idx]), (boxC[idx]+100, boxR[idx]+100), (0, 0, 0))
                    cv2.imshow("test", img)
                    cv2.waitKey()'''

        return box
        # print(MatchTable.shape)

    def ToTargetImageFormat(self, img):
        img = LabelImage.ToTargetImageFormat(img, self.ImageInfo)
        return img

    def DataSetPredict(self, source):
        imgs = source.GetImageData(self.ImageInfo, Dim=4)
        labels, _ = source.ToIntLable(ArrayExtend=True)
        self.KerasMdl.evaluate(imgs, labels)
        return 1

    def ImagePredict(self, source):
        source = self.ToTargetImageFormat(source)
        # extend to four dimemsion array
        source = source.reshape((1,) + source.shape+(1,))
        return self.KerasMdl.predict(source)

    def Evaluate(self, source):
        if isinstance(source, LabelImage.DataObj):
            result = self.DataSetPredict(source)
            return result
        elif isinstance(source, str):
            source = cv2.imread(source)
            return self.ImagePredict(source)

    def Predict(self, source):
        imgs = source.GetImageData(self.ImageInfo, Dim=4)
        labels, _ = source.ToIntLable(ArrayExtend=True)
        return self.KerasMdl.predict(imgs)

    def LoadModelFromTxt(self, txtFile):
        with open(txtFile, 'r') as f:
            lines = f.readlines()
        

        for l in lines:
            try:
                l = l.strip('\n')
                if l != '' and l[0]!='#':
                    target = l
                    exec(target)
                    self.LayerNum = self.LayerNum+1
            except OSError as err:
                print("Error on adding ", str(err))
                pass
        
        exec('self.KerasMdl = model') 

    def AssignModel(self,mdl):
        self.KerasMdl = mdl