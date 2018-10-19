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

fullData = c1.CreateILBPNet()

# time test
if TimeTest == True:
    for i in range(0, 100):
        start = time.perf_counter()
        firstSel = c1.CreateILBPNet()
        end = time.perf_counter()
        elapsed = end - start
        print("elapsed time = {:.12f} seconds".format(elapsed))

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
