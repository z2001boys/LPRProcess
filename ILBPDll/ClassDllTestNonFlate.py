import CudaComMat
import cv2
import numpy
import matplotlib.pyplot as plt
import math
import CommonMath
import time

TimeTest = True

img1 = cv2.imread('D:\\test.jpg', 0)
# c1 = CudaComMat.CudaMat(ReadPath='D:\\3d\\Calib0827\\0\\cam0.png')
c1 = CudaComMat.CudaMat(img=img1)
c1.Upload()

firstSel,maxSel = c1.CreateILBP(Flatten=False)

plt.imshow(img1)
# x,y
for i in range(0, c1.height()):
    for j in range(0, c1.width()):
        for r in range(8):
            if firstSel[i,j,r]>=1:
                row, col = CommonMath.GetArrow(i, j, r, 1)
                plt.annotate('', xy=(col, row), xycoords='data',
                    xytext = (j, i), textcoords = 'data',
                    arrowprops = dict(facecolor='green', arrowstyle='->', linewidth='1'))


plt.show()
