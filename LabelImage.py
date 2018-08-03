

import numpy as np
import os
import cv2
import progressbar
import ImageInfoClass


def IsImage(s):
    if ("png" in s or "bmp" in s or "jpg" in s) and os.path.exists(s):
        return True
    else:
        return False

def NumpyArrayToGray(NpArray):
    grayData = np.zeros([NpArray.shape[0],NpArray.shape[1],NpArray.shape[2]])
    for i in range(len(NpArray)):
        grayData[i,:,:] = np.dot(NpArray[i,:,:,:],[0.299, 0.587, 0.114])
    if len(NpArray.shape)==4:
        grayData = grayData[:,:,:,np.newaxis]
    return grayData

#untested
def ExtendToColor(img):
    Data = np.zeros([img[0],img[1],3])
    Data[:,:,1]=img
    Data[:,:,2]=img
    Data[:,:,3]=img

def ToGray(img):
    img = np.dot(img,[0.33,0.33,0.33])
    return img



class DataObj:
    def __init__(self):
        self.imgWidth = 100
        self.imgHeight = 100
        self.IsGray = False
        self.ImageArray = []
        self.Label = []

    def GetImageData(self,ImgInfo, type='numpy',Dim = 3) :
        
        #basic info
        imgNum = self.ImageArray.shape[0]
        FlattenSize = ImgInfo.Size[0] * ImgInfo.Size[1] * ImgInfo.Channel

        print("Preprocessing images...")
        print("==Target Image Size : ",ImgInfo.Size)
        print("==Target Image Channel : ",ImgInfo.Channel)
        print("==Target Image Flatten : ",ImgInfo.NeedFlatten)


        #create return mat
        if ImgInfo.NeedFlatten:
            NewImgData = np.zeros([self.ImageArray.shape[0], ImgInfo.Size[0]*ImgInfo.Size[1]])
        else:
            NewImgData = np.zeros([self.ImageArray.shape[0], ImgInfo.Size[0],ImgInfo.Size[1]])

        if len(self.ImageArray.shape)==4 and self.ImageArray.shape[3]==3:
            oriDataIsGray = False
        else:
            oriDataIsGray = True

        
        
        #add to result array
        for i in range(imgNum):
            img = self.ImageArray[i,]

            #color option
            if oriDataIsGray and ImgInfo.Channel == 3:
                img = ExtendToColor(img)
            
            if not oriDataIsGray and ImgInfo.Channel == 1:
                img = ToGray(img)
            
            img = cv2.resize(img,(ImgInfo.Size[0],ImgInfo.Size[1]))

            #flatten option
            if ImgInfo.NeedFlatten:
                img = img.reshape([FlattenSize])

            NewImgData[i,:] = img
        # 檢查是否為四軸
        if Dim==4 and len(NewImgData.shape)==3:
            NewImgData = NewImgData[:,:,:,np.newaxis]

        #回傳結果
        return NewImgData

    # 讀取 list中的影像 並且存到物件中

    def LoadFromList(self, ListName, TrimName=""):
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
            singleLine = singleLine.strip(TrimName)
            info = singleLine.split(',')

            # load single image
            if len(info) == 2 and IsImage(info[0]):
                if self.IsGray == True:
                    img = cv2.imread(info[0], 0)
                else:
                    img = cv2.imread(info[0])

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
    def ToIntLable(self, ArrayExtend=False):
        print("Preprocessing labels...")

        localLabel = self.Label
        existDict = dict()
        dataSize = len(localLabel)
        intLable = [0] * dataSize

        # 使用字典擴展成label
        for i in range(0, len(localLabel)):
            if localLabel[i] in existDict:
                intLable[i] = existDict[localLabel[i]]
            else:
                existDict[localLabel[i]] = len(existDict)

        # 如果不需要擴展成矩陣則直接回傳，回傳內容為[0 1 2 3 4...]
        if ArrayExtend == False:
            return intLable, dataSize

        # 如果需要擴展成矩陣則為下，擴展內容為 lable0 = [1 0 0 0...]
        #                                   label1 = [0 1 0 0...]
        # gen data
        labelSize = len(existDict)
        ExtendLabel = np.zeros((dataSize, len(existDict)))

        preLabel = np.zeros((labelSize, labelSize))
        for i in range(0, labelSize):
            preLabel[i, i] = 1

        for i in range(0, dataSize):
            ExtendLabel[i, :] = preLabel[existDict[localLabel[i]], :]

        return ExtendLabel, labelSize
