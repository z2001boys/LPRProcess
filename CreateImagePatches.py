import cv2
import numpy as np
import math


def CreatePatch(img,
                TargetSize=(100, 100),
                Stride=(2, 2)):

    sourceSize = img.shape

    trimSizeR = sourceSize[0]-TargetSize[0]+1
    trimSizeC = sourceSize[1]-TargetSize[1]+1

    sizeOfR = math.ceil(trimSizeR/Stride[0])
    sizeOfC = math.ceil(trimSizeC/Stride[1])
    totalSize = sizeOfR*sizeOfC

    result = np.zeros(
        (totalSize, TargetSize[0], TargetSize[1], 1), dtype=float)

    # calc start position

    c1 = np.tile(np.arange(0, trimSizeC, Stride[1]), sizeOfC)
    c2 = np.tile(c1 + TargetSize[1], sizeOfR)
    rSample = np.arange(0, trimSizeR, Stride[0])
    r1 = np.zeros((totalSize, ), dtype=int)
    r2 = np.zeros((totalSize, ), dtype=int)

    for i in range(sizeOfR):
        r = [i*sizeOfC, (i+1)*sizeOfC]
        r1[r[0]: r[1]] = np.repeat(rSample[i], sizeOfC)
        r2[r[0]: r[1]] = r1[r[0]: r[1]] + TargetSize[1]

    for i in range(totalSize):
        result[i, :, :, 0] = img[r1[i]: r2[i], c1[i]: c2[i]]
        #cv2.imshow('test', result[i, :, :])
        # cv2.waitKey()

    return result, (sizeOfR, sizeOfC), r1, c1


# test
img = cv2.imread('FullImage/1.png', 0)
fullImg = CreatePatch(img, Stride=(30, 30))
