import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras import backend as K
import LabelImage
import ImageInfoClass as IF
import CreateImagePatches as CIP
import numpy as np
import cv2
import CommonMath
import CommonStruct
import gc

'''keras import'''
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD

from CNN_Module import conv_bn_relu, basic_block, bottleneck, bn_relu, inception_block_v1,darknet_bottleneck, conv_block, dense_block, transition_block, mn_conv_block, depthwise_conv_block, inverted_res_block, make_divisible, correct_pad, stem_v4, inception_a_v4, inception_b_v4, inception_c_v4, reduction_a_v4, reduction_b_v4, stem_inception_resnet_v1, inception_resnet_a_v1, inception_resnet_b_v1, inception_resnet_c_v1, reduction_resnet_b_v1, stem_inception_resnet_v2, inception_resnet_a_v2, inception_resnet_b_v2, inception_resnet_c_v2, reduction_resnet_a, reduction_resnet_b_v2, grouped_conv_block, xt_bottleneck_block, grouped_conv_block_v2, xt_bottleneck_block_v2, separable_conv_block, adjust_block, normal_cell, reduction_cell, NasNet

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
            _loadPath+_modelName)

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

    def Train(self, imgObj, batch_size=-1, epochs=-1, verbose=-1, callbacks=None):
        print("Prepare data...")
        imageData = imgObj.GetImageData(self.ImageInfo, Dim=4)
        label, _ = imgObj.ToIntLable(FinalTarget = "ArrayExtend")

        #print("Start Training")
        #print("==Total layer : ", self.LayerNum)

        # build train command
        '''
        targetStr = 'self.KerasMdl.fit(imageData, label'
        if(batch_size != -1):
            targetStr = targetStr + ',batch_size='+str(batch_size)
        if(epochs != -1):
            targetStr = targetStr + ',epochs='+str(epochs)
        if(verbose != -1):
            targetStr = targetStr + ',verbose='+str(verbose)

        targetStr = targetStr+')'

        eval(targetStr)
        '''
        # v = self.KerasMdl.predict(imageData)
        targetStr = 'self.KerasMdl.fit(imageData, label, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)'
        eval(targetStr)

    def TrainByGener(self, trainPath,
                     IncludeSet=[],
                     GlobalEpochs=10,
                     GlobalBatchSize=3000,
                     batch_size=-1, epochs=-1, verbose=-1):
        if IncludeSet == []:
            raise Exception('Data Set cant be empty')

        imgObj = LabelImage.DataObj()
        labels, imgPaths = imgObj.CreateAccessCache(trainPath, IncludeSet)
        baseEpoch = 10

        for gi in range(GlobalEpochs):
            imgObj.GenImage(
                imgPaths, labels, IncludeSet, self.ImageInfo, eachSize=(400, 600))
            imgObj.LoadFromList(
                'D:/tempLabel.txt', (self.ImageInfo.Size[0], self.ImageInfo.Size[1]), IncludeSet)

            # get learning data
            x_train = imgObj.GetImageData(self.ImageInfo, Dim=4)
            y_train, _ = imgObj.ToIntLable(ArrayExtend=True)

            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            datagen.fit(x_train)

            self.KerasMdl.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                                        steps_per_epoch=baseEpoch, epochs=epochs)

            # prepare for next round
            baseEpoch = baseEpoch+5

            gc.collect()
            continue  # for next round

        return 1

    def Compile(self, _optimize, _loss, _metrics):
        self.KerasMdl.compile(optimizer=_optimize,
                              loss=_loss,
                              metrics=_metrics)

    def FindInImage(self, img, score, labels):
        imgPatch, BoxSize, boxR, boxC = CIP.CreatePatch(
            img, Stride=(10, 10), TargetSize=(64, 64))
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
        '''
        with open(txtFile, 'r') as f:
            lines = f.readlines()

        #prevStr = 'self.KerasMdl.add('

        for l in lines:
            try:
                l = l.strip('\n')
                if l != '':
                    #target = prevStr+l+')'
                    #eval(target)
                    exec(l)
                    self.LayerNum = self.LayerNum+1
            except OSError as err:
                print("Error on adding ", str(err))
                pass
        
        exec('self.KerasMdl = model')
        
        '''
        with open(txtFile, 'r') as f:
            lines = f.read()
        exec(lines)
        exec('self.KerasMdl = model')        
        
    # ResNet Module
    '''
    def bn_relu(self, input):
        bn = BatchNormalization(axis=3)(input)
        return Activation("relu")(bn)

    def conv_bn_relu(self, **conv_params):
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

        def f(input):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(input)
            return self.bn_relu(conv)
        return f

    def bn_relu_conv(self, **conv_params):
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            activation = self.bn_relu(input)
            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(activation)
        return f

    def shortcut(self, input, residual):
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input

        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        return add([shortcut, residual])

    def basic_block(self, filters, init_strides=(1, 1), first_layer_first_block=False):
        def f(input):
            if first_layer_first_block:
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides, padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l1(1e-4))(input)
            else:
                conv1 = self.bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                     strides=init_strides)(input)

            residual = self.bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return self.shortcut(input, residual)

        return f

    def bottleneck(self, filters, init_strides=(1, 1), first_layer_first_block=False):
        def f(input):
            if first_layer_first_block:
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=init_strides, padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4))(input)
            else:
                conv_1_1 = self.bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                        strides=init_strides)(input)

            conv_3_3 = self.bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
            residual = self.bn_relu_conv(filters=filters*4, kernel_size=(1, 1))(conv_3_3)
            return self.shortcut(input, residual)

        return f
    '''
