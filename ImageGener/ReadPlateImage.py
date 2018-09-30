import cv2
import re
import numpy
import math
from random import randint


def IntersectionArea(r11, c11, r12, c12, r21, c21, r22, c22):
    x_overlap = max(0, min(
        c12, c22) - max(c11, c21))
    y_overlap = max(0, min(
        r12, r22) - max(r11, r21))
    overlapArea = x_overlap * y_overlap
    return overlapArea


def GetROI(info):
    # roi = [x1,y1,x2,y2]
    roiInfo = re.findall("\((\d+),(\d+),(\d+),(\d+)\)", info)
    roi = numpy.zeros((len(roiInfo), 4))
    for i in range(len(roiInfo)):
        coordinate = roiInfo[i]
        roi[i, :] = (int(coordinate[0]), int(coordinate[1]),
                     int(coordinate[2]), int(coordinate[3]))
    return roi


def RandROI(imgH, imgW, minBound, maxBound):
    r1 = randint(0, imgH-1)
    c1 = randint(0, imgW-1)
    boxSize = randint(minBound, maxBound)
    r2 = r1+boxSize
    c2 = c1+boxSize
    return numpy.asarray((r1, c1, r2, c2))


def NoOverLapWithROI(rois, roi):
    for i in range(len(rois)):
        BaseArea = (rois[i, 2]-rois[i, 0])*(rois[i, 3]-rois[i, 1])
        interArea = IntersectionArea(
            rois[i, 1], rois[i, 0], rois[i, 3], rois[i, 2], roi[0], roi[1], roi[2], roi[3])
        if interArea > BaseArea*0.4:
            return True

    return False


def ReadPlateImg(filePath):
    with open(filePath) as f:
        lines = f.readlines()

    plateCount = 0
    nonePlateCount = 0

    for l in lines:
        s = l.split(';')
        fileName = s[0]
        roi = GetROI(s[1])
        if roi.shape[0] == 0:
            continue

        img = cv2.imread(fileName)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        for i in range(roi.shape[0]):
            height = roi[i, 3]-roi[i, 1]
            width = roi[i, 2]-roi[i, 0]
            centerR = int((roi[i, 3]+roi[i, 1])/2)
            centerC = int((roi[i, 2]+roi[i, 0])/2)
            boxSize = int(max(width, height)/2)
            subImg = img[centerR-boxSize:centerR +
                         boxSize, centerC-boxSize:centerC+boxSize, :]
            cv2.imwrite("LicensePlateDataSet\\Plates\\" +
                        str(plateCount)+".jpg", subImg)
            plateCount = plateCount+1
        # extract non plate image
        if roi.shape[0] != 0:
            for i in range(1000):
                randRoi = RandROI(imgHeight, imgWidth, 100, 600)
                while NoOverLapWithROI(roi, randRoi) or randRoi[2] > imgHeight or randRoi[3] > imgWidth:
                    # repick
                    randRoi = RandROI(imgHeight, imgWidth, 100, 600)
                subImg = img[randRoi[0]:randRoi[2], randRoi[1]:randRoi[3]]
                cv2.imwrite("LicensePlateDataSet\\Non_Plates\\" +
                            str(nonePlateCount)+".jpg", subImg)
                nonePlateCount = nonePlateCount+1


v = ReadPlateImg('C:\\Users\\egg_yang\\Documents\\plate.txt')
