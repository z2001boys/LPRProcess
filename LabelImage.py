

import numpy as np
import os
import cv2
import progressbar
import ImageInfoClass
import tensorflow as tf
from random import randint


def IsImage(s):
    if ("png" in s or "bmp" in s or "jpg" in s):
        return True
    else:
        return False


def PathCheck(s):
    if s[len(s)-1] != '/':
        s = s + '/'
    return s


def ToTargetImageFormat(img, ImgInfo, oriDataIsGray=-1, FlattenSize=-1):

    if oriDataIsGray == -1:
        if len(img.shape) == 3 and img.shape[2] == 3:
            oriDataIsGray = False
        else:
            oriDataIsGray = True

    if FlattenSize == -1:
        FlattenSize = ImgInfo.Size[0] * ImgInfo.Size[1] * ImgInfo.Channel

    if oriDataIsGray and ImgInfo.Channel == 3:
        img = ExtendToColor(img)

    if not oriDataIsGray and ImgInfo.Channel == 1:
        img = ToGray(img)

    if img.shape[0] != ImgInfo.Size[0] or img.shape[1] != ImgInfo.Size[1]:
        img = cv2.resize(img, (ImgInfo.Size[0], ImgInfo.Size[1]))

    # flatten option
    if ImgInfo.NeedFlatten:
        img = img.reshape([FlattenSize])

    if img.dtype == np.uint8:
        img = img.astype('float')

    return img


def NumpyArrayToGray(NpArray):
    grayData = np.zeros([NpArray.shape[0], NpArray.shape[1], NpArray.shape[2]])
    for i in range(len(NpArray)):
        grayData[i, :, :] = np.dot(NpArray[i, :, :, :], [0.299, 0.587, 0.114])
    if len(NpArray.shape) == 4:
        grayData = grayData[:, :, :, np.newaxis]
    return grayData

# untested


def ExtendToColor(img):
    Data = np.zeros([img[0], img[1], 3])
    Data[:, :, 1] = img
    Data[:, :, 2] = img
    Data[:, :, 3] = img


def ToGray(img):
    img = np.dot(img, [0.33, 0.33, 0.33])
    return img


class DictListObj:
    def __init__(self):
        self.ListObj = []


class DataObj:
    def __init__(self):
        self.imgWidth = 100
        self.imgHeight = 100
        self.IsGray = False
        self.ImageArray = []
        self.MappedClass = []
        self.Label = []

    def CreateAccessCache(self, path, wantedLabel):
        path = PathCheck(path)
        fileList = os.listdir(path)
        resultPath = []
        resultLabel = []
        for f in fileList:
            if f in fileList:
                imgs = os.listdir(path+f)
                for i in imgs:
                    if IsImage(i):
                        resultPath.append(path+f+'/'+i)
                        resultLabel.append(f)

        return resultLabel, resultPath

    def GenImage(self, imgPaths, labels, preSet, imgInfo, eachSize):
        if len(eachSize) != len(preSet):
            raise Exception('Data Set number not match')

        labelDict = dict()
        for l in preSet:
            if not l in labelDict:
                newObj = DictListObj()
                labelDict[l] = newObj

        # create map index
        for i in range(len(labels)):
            labelDict[labels[i]].ListObj.append(i)

        # create labels
        intLabel = []
        for i in range(len(eachSize)):
            for j in range(eachSize[i]):
                intLabel.append(i)

        resultPath = []
        for i in range(len(preSet)):
            for j in range(eachSize[i]):
                idx = randint(0, len(labelDict[preSet[i]].ListObj)-1)
                selected = labelDict[preSet[i]].ListObj[idx]
                resultPath.append(imgPaths[selected])

        with open("D:\\tempLabel.txt", 'w') as f:
            for i in range(len(resultPath)):
                f.write(resultPath[i]+','+preSet[intLabel[i]]+'\n')

    def GetImageData(self, ImgInfo, type='numpy', Dim=3):

        # basic info
        imgNum = self.ImageArray.shape[0]
        FlattenSize = ImgInfo.Size[0] * ImgInfo.Size[1] * ImgInfo.Channel

        print("Preprocessing images...")
        print("==Target Image Size : ", ImgInfo.Size)
        print("==Target Image Channel : ", ImgInfo.Channel)
        print("==Target Image Flatten : ", ImgInfo.NeedFlatten)

        # create return mat
        if ImgInfo.NeedFlatten:
            NewImgData = np.zeros(
                [self.ImageArray.shape[0], ImgInfo.Size[0]*ImgInfo.Size[1]])
        else:
            NewImgData = np.zeros(
                [self.ImageArray.shape[0], ImgInfo.Size[0], ImgInfo.Size[1], ImgInfo.Channel], dtype=np.float)

        if len(self.ImageArray.shape) == 4 and self.ImageArray.shape[3] == 3:
            oriDataIsGray = False
        else:
            oriDataIsGray = True

        # add to result array
        for i in range(imgNum):
            img = self.ImageArray[i, ]

            img = ToTargetImageFormat(
                img, ImgInfo, oriDataIsGray=oriDataIsGray, FlattenSize=FlattenSize)

            if len(NewImgData.shape) == 4 and len(img.shape) == 2:
                img = img[:, :, np.newaxis]

            NewImgData[i, :] = img
        # 檢查是否為四軸
        if Dim == 4 and len(NewImgData.shape) == 3:
            NewImgData = NewImgData[:, :, :, np.newaxis]

        # 回傳結果
        return NewImgData

    def GetSingleImage(self, idx):
        return self.ImageArray[idx, :], self.Label[idx]

    # 讀取 list中的影像 並且存到物件中
    # sorted class 為預先排好的class名稱
    def LoadFromList(self, ListName, TargetSize=(100, 100), SortedClass=[], TrimName=""):
        self.MappedClass = SortedClass

        # read lines in file
        with open(ListName, 'r') as file:
            content = file.readlines()
        content = [x.strip() for x in content]

        # 準備讀取
        current = 0
        totalCount = len(content)
        print("{} images prepare to load...".format(totalCount))
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        # 開始讀取
        self.ImageArray = []
        self.Label = []
        for singleLine in content:
            singleLine = singleLine.replace(TrimName, '')
            info = singleLine.split(',')

            # load single image
            if len(info) == 2 and IsImage(info[0]):
                if self.IsGray == True:
                    img = cv2.imread(info[0], 0)
                else:
                    img = cv2.imread(info[0])

                img = cv2.resize(img, TargetSize)

                # Concat images
                if self.IsGray:
                    self.ImageArray.append(np.asarray(img[:, :]))
                else:
                    self.ImageArray.append(np.asarray(img[:, :, :]))

                # Concat labels
                self.Label.append(info[1])

            # bar progress
            barPos = current/totalCount*100
            bar.update(barPos)
            current = current+1
        # end process
        bar.finish()
        self.ImageArray = np.asarray(self.ImageArray)

    # 將label 轉成數字
    # FinalTarget = ArrayExtend, BinaryClass
    def ToIntLable(self, FinalTarget = ""):
        print("Preprocessing labels...")

        localLabel = self.Label
        existDict = dict()
        dataSize = len(localLabel)
        intLable = [0] * dataSize

        if self.MappedClass != []:
            for i in range(len(self.MappedClass)):
                if self.MappedClass[i] not in existDict:
                    existDict[self.MappedClass[i]] = len(existDict)

        # 使用字典擴展成label
        for i in range(0, len(localLabel)):
            if localLabel[i] in existDict:
                intLable[i] = existDict[localLabel[i]]
            else:
                existDict[localLabel[i]] = len(existDict)

        # 如果不需要擴展成矩陣則直接回傳，回傳內容為[0 1 2 3 4...]
        if FinalTarget=="":
            return intLable, dataSize

        if FinalTarget == "ArrayExtend":
            # 如果需要擴展成矩陣則為下，擴展內容為 lable0 = [1 0 0 0...]
            #                                   label1 = [0 1 0 0...]
            # func map to
            # keras.utils.to_categorical

            #extendLabel = tf.keras.utils.to_categorical(intLable)
            # return extendLabel, len(existDict)
            
            labelSize = len(existDict)
            ExtendLabel = np.zeros((dataSize, len(existDict)))

            preLabel = np.zeros((labelSize, labelSize))
            for i in range(0, labelSize):
                preLabel[i, i] = 1

            for i in range(0, dataSize):
                ExtendLabel[i, :] = preLabel[existDict[localLabel[i]], :]

            return ExtendLabel, labelSize
        
        if FinalTarget == "BinaryClass":
            ExtendLabel = np.zeros((dataSize,1))

            for i in range(0,dataSize):
                if existDict[localLabel[i]]==0:
                    ExtendLabel[i][0] = 1
                else:
                    ExtendLabel[i][0] = -1

            return ExtendLabel,2