

import numpy as np
import os
import cv2
import progressbar
import ImageInfoClass
import tensorflow as tf
from random import randint
import ILBPDll.CudaComMat as CuMat
import matplotlib.pyplot as plt
import numpy
import gc
import random
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import BlockBuilder

def IsImage(s):
    if ("png" in s or "bmp" in s or "jpg" in s):
        return True
    else:
        return False


def PathCheck(s):
    if s[len(s)-1] != '/':
        s = s + '/'
    return s


def ToTargetImageFormat(img, ImgInfo, FlattenSize=-1,Format = numpy.float,Rotation=0,Noise=0):

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


    if Rotation == 'rdn':
        rot = randint(0,359)
        M = cv2.getRotationMatrix2D((ImgInfo.Size[0]/2, ImgInfo.Size[1]/2),rot,1)
        img = cv2.warpAffine(img,M,(ImgInfo.Size[0], ImgInfo.Size[1]))
    elif Rotation != 0 :        
        M = cv2.getRotationMatrix2D((ImgInfo.Size[0]/2, ImgInfo.Size[1]/2),Rotation,1)
        img = cv2.warpAffine(img,M,(ImgInfo.Size[0], ImgInfo.Size[1]))    
    

    # flatten option
    if ImgInfo.NeedFlatten:
        img = img.reshape([FlattenSize])

    if img.dtype != Format:
        img = img.astype(Format)

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
        self.ImgList = []
        self.Rotation = 0
        self.AddNoies =0

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

    def CreateAccessCacheByFile(self,fileName,wantedLabel,TrimName=''):
        # read lines in file
        with open(fileName, 'r') as file:
            content = file.readlines()
        content = [x.strip() for x in content]

        resultPath = []
        resultLabel = []

        for l in content:
            splitData = l.split(',')
            if any(splitData[1] in s for s in wantedLabel):
                resultLabel.append(splitData[1])
                resultPath.append(splitData[0].replace(TrimName, ''))
                

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

    def GetListSize(self):
        return len(self.ImgList)

    def BuildHOG(self,imgArr,cellSize=8,cellNum=3,oriNum=10):
        imgNum = imgArr.shape[0]
        singleFeature = hog(imgArr[0,:,:,:],
            pixels_per_cell=(cellSize,cellSize),
            cells_per_block=(cellNum,cellNum),
            orientations=oriNum,
            block_norm='L2-Hys',)
        featureDim = singleFeature.shape[0]
        hogData = np.zeros((imgNum,featureDim))
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for j in range(imgNum):
            hogData[j,:] = hog(imgArr[j,:,:,:],
                pixels_per_cell=(cellSize,cellSize),
                cells_per_block=(cellNum,cellNum),
                orientations=oriNum,
                block_norm='L2-Hys')
            bar.update(j*100/imgNum)
        bar.finish()
        gc.collect()
        return hogData

    def SetSize(self):
        return len(self.ImgList)

    def GenAugData(self,imgInfo,times=10,PickSize = 1000,PreProcess=''):


        dataGen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.2,
                height_shift_range=0.2,                
            )
        count = 0;

        listPath = 'D:\\temp\\tmpList.txt'

        print('data aug...')
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        with open(listPath,'w') as f:
            for i in range(len(self.ImgList)):
                path = self.ImgList[i]
                l = self.Label[i]
                img = cv2.imread(path)
                imgSize = img.shape
                chExt = False
                if len(imgSize)==3:
                    chExt = True
                if chExt == True:
                    data = dataGen.flow(img.reshape(-1,imgSize[0],imgSize[1],3),numpy.ones(1),batch_size=1)
                else:    
                    data = dataGen.flow(img.reshape(-1,imgSize[0],imgSize[1],1),numpy.ones(1),batch_size=1)
                for j,newImgs in enumerate(data):                
                    fileName = 'D:\\temp\\tmpImg\\'+str(count)+'.png';                    
                    cv2.imwrite(fileName,newImgs[0][0,:,:,:])                    
                    f.write(fileName+','+str(l)+'\n')        
                    count = count +1 
                    if j >= times:
                        break
                bar.update(i*100/len(self.ImgList))
        bar.finish()

        tmpObj = DataObj()
        tmpObj.LoadList(listPath,SortedClass=self.MappedClass)        
        imgOjb = tmpObj.RadomLoad(imgInfo,PickSize=-1,PreProcess=PreProcess)

        gc.collect()
        return imgOjb

    def BuildLBP(self,imgArr,cellSize=8,cellNum=3):
        
        imgNum = imgArr.shape[0]
        firstFeaure = BlockBuilder.buildHist(imgArr[0,:,:,0],
            cellSize = 10,
            blockSize=2)
        maxFeature = BlockBuilder.buildHist(imgArr[0,:,:,0],
            cellSize = 20,
            blockSize=3)
        featureDim1 = firstFeaure.shape[0]
        featureDim2 = maxFeature.shape[0]
        hogData = np.zeros((imgNum,featureDim1+featureDim2))
        print('----------Start to generate feature-----------------')
        print('feature num1:',featureDim1)
        print('feature num2:',featureDim2)
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for j in range(imgNum):
            hogData[j,0:featureDim1] = BlockBuilder.buildHist(imgArr[j,:,:,0],
            cellSize = 10,
            blockSize=2,
            clip=0.3)
            bar.update(j*100/imgNum)
            hogData[j,featureDim1:featureDim1+featureDim2] = BlockBuilder.buildHist(imgArr[j,:,:,1],
            cellSize = 20,
            blockSize=3,
            clip=0.4)
            bar.update(j*100/imgNum)
        bar.finish()
        gc.collect()
        return hogData

    

    def RadomLoad(self, ImgInfo , type='numpy', Dim=3,PreProcess = 'none',PickSize = 1000,randIdx=[],
        kerasLabel = True,
        SeqLoad = False,
        featurePlaceHolder = 0,
        dataAug = False ):

        if( dataAug==True ):
            return self.GenAugData(times=10,imgInfo=ImgInfo,PickSize=PickSize,PreProcess=PreProcess)

        # basic info
        totalSize = len(self.ImgList)
        if PickSize == -1:
            PickSize = len(self.ImgList)
        imgNum = min(len(self.ImgList),PickSize)
        FlattenSize = ImgInfo.Size[0] * ImgInfo.Size[1] * ImgInfo.Channel
        print("Preprocessing images...")
        print("==Target Image Size : ", ImgInfo.Size)
        print("==Target Image Channel : ", ImgInfo.Channel)
        print("==Target Image Flatten : ", ImgInfo.NeedFlatten)

        if randIdx==[]:
            print("==No specific index select by LBImg, Radom Loading = ",SeqLoad)
            if SeqLoad == True:                
                randIdx = numpy.arange(0,min(totalSize,imgNum),1)
            else:                
                randIdx = random.sample(range(totalSize), min(totalSize,imgNum))

        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        FlattenSize = ImgInfo.Size[0]*ImgInfo.Size[1]

        imgChannel = ImgInfo.Channel
        if PreProcess =="ILBPNet":
            imgChannel = 3
        if PreProcess == "ILBPNet+":
            ImgInfo.NeedFlatten = True
            virImg = numpy.zeros([ImgInfo.Size[0],ImgInfo.Size[1]])
            flbpSize = BlockBuilder.buildHist(virImg,cellSize=10,blockSize=2).shape[0]
            mlbpSize = BlockBuilder.buildHist(virImg,cellSize=15,blockSize=3).shape[0]
            FlattenSize = 10000 + flbpSize+mlbpSize

        # create return mat
        if ImgInfo.NeedFlatten:
            NewImgData = np.zeros(
                [imgNum, FlattenSize])
        else:
            NewImgData = np.zeros(
                [imgNum, ImgInfo.Size[0], ImgInfo.Size[1], imgChannel], dtype=np.float)

        curLabel = []


        # add to result array
        for i in range(imgNum):
            idx = randIdx[i]
            img = cv2.imread(self.ImgList[idx])
            curLabel.append(self.Label[idx])

            f = numpy.float

            if PreProcess == "FeatureUnion":
                feature = self.BuildLBP(img)
                NewImgData[i, featurePlaceHolder:featurePlaceHolder+FlattenSize] = feature
                bar.update(i*100/imgNum)
            if PreProcess == "ILBPNet+":
                img = ToTargetImageFormat(
                    img, ImgInfo, FlattenSize=-1, Format = f,Rotation=self.Rotation)
                gpuMat = gpuMat = CuMat.CudaMat(img=img.reshape(100,100))
                ilbpImg = gpuMat.CreateILBP(Flatten=True).astype(np.float)
                NewImgData[i, 0:10000] = img[:]
                flbp = BlockBuilder.buildHist(ilbpImg[:,:,0],cellSize=10,blockSize=2)
                mlbp = BlockBuilder.buildHist(ilbpImg[:,:,1],cellSize=15,blockSize=3)
                NewImgData[i,10000:10000+flbpSize] = flbp
                NewImgData[i,10000+flbpSize:10000+flbpSize+mlbpSize] = mlbp
                bar.update(i*100/imgNum)
            else:
                if PreProcess == "ILBPNet":
                    f = numpy.uint8

                img = ToTargetImageFormat(
                    img, ImgInfo, FlattenSize=-1, Format = f,Rotation=self.Rotation)
            

                if PreProcess == "ILBPNet":
                    gpuMat = CuMat.CudaMat(img=img)
                    #img = gpuMat.CreateILBPNet().astype(np.float)
                    img = gpuMat.CreateILBP(Flatten=True).astype(np.float)


                if len(NewImgData.shape) == 4 and len(img.shape) == 2:
                    img = img[:, :, np.newaxis]

                if i%1000 == 0:
                    gc.collect()

                NewImgData[i, :] = img
                bar.update(i*100/imgNum)

        # 檢查是否為四軸
        if Dim == 4 and len(NewImgData.shape) == 3:
            NewImgData = NewImgData[:, :, :, np.newaxis]
        if Dim == 3 and len(NewImgData.shape) == 2:
            NewImgData = NewImgData[:, : , np.newaxis]
        bar.finish()

        gc.collect()

        if kerasLabel == True:
            labels,_ = self.ToIntLable(curLabel,FinalTarget="ArrayExtend")
        else:
            labels,_ = self.ToIntLable(curLabel)
        



        # 回傳結果
        return NewImgData,labels



    def GetImageData(self, ImgInfo, type='numpy', Dim=3,PreProcess = 'none'):

        # basic info
        imgNum = self.ImageArray.shape[0]
        FlattenSize = ImgInfo.Size[0] * ImgInfo.Size[1] * ImgInfo.Channel

        print("Preprocessing images...")
        print("==Target Image Size : ", ImgInfo.Size)
        print("==Target Image Channel : ", ImgInfo.Channel)
        print("==Target Image Flatten : ", ImgInfo.NeedFlatten)

        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        imgChannel = ImgInfo.Channel
        if PreProcess =="ILBP":
            imgChannel = 16

        # create return mat
        if ImgInfo.NeedFlatten:
            NewImgData = np.zeros(
                [self.ImageArray.shape[0], ImgInfo.Size[0]*ImgInfo.Size[1]])
        else:
            NewImgData = np.zeros(
                [self.ImageArray.shape[0], ImgInfo.Size[0], ImgInfo.Size[1], imgChannel], dtype=np.float)

        if len(self.ImageArray.shape) == 4 and self.ImageArray.shape[3] == 3:
            oriDataIsGray = False
        else:
            oriDataIsGray = True

        # add to result array
        for i in range(imgNum):
            img = self.ImageArray[i, ]

            f = numpy.float

            if PreProcess == "ILBP":
                f = numpy.uint8

            img = ToTargetImageFormat(
                img, ImgInfo, oriDataIsGray=oriDataIsGray, FlattenSize=FlattenSize, Format = f)

            if PreProcess == "ILBP":
                gpuMat = CuMat.CudaMat(img=img)
                img = gpuMat.CreateILBPNet().astype(np.float)


            if len(NewImgData.shape) == 4 and len(img.shape) == 2:
                img = img[:, :, np.newaxis]

            if i%1000 == 0:
                gc.collect()

            NewImgData[i, :] = img
            bar.update(i*100/imgNum)

        # 檢查是否為四軸
        if Dim == 4 and len(NewImgData.shape) == 3:
            NewImgData = NewImgData[:, :, :, np.newaxis]
        bar.finish()
        # 回傳結果
        return NewImgData

    def GetSingleImage(self, idx):
        return self.ImageArray[idx, :], self.Label[idx]

     # 讀取 list中的"路徑" 並且存到物件中
    # sorted class 為預先排好的class名稱
    def LoadList(self, ListName, TargetSize=(100, 100), SortedClass=[], TrimName=""):
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

        for singleLine in content:
            singleLine = singleLine.replace(TrimName, '')
            info = singleLine.split(',')

            
            # load single image
            if len(info) == 2 and IsImage(info[0]) :
                if  any(info[1] in s for s in SortedClass):
                    self.ImgList.append(info[0])

                    # Concat labels
                    self.Label.append(info[1])

            # bar progress
            barPos = current/totalCount*100
            bar.update(barPos)
            current = current+1
        # end process
        bar.finish()
        self.ImageArray = []



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
    def ToIntLable(self,UseLabel=[], FinalTarget = ""):
        print("Preprocessing labels...")

        localLabel = UseLabel
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
            ExtendLabel = np.zeros((dataSize, len(existDict)),dtype=np.float)

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