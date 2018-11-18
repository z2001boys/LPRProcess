import CudaComMat
import cv2
import numpy
import matplotlib.pyplot as plt
import math
import CommonMath
import time
import Noiser

TimeTest = False

img1 = cv2.imread('D:\\TestImg\\test7.jpg', 0)
img1 = cv2.resize(img1,(100,100))
#img1 = Noiser.noisy("s&p",img1)
# c1 = CudaComMat.CudaMat(ReadPath='D:\\3d\\Calib0827\\0\\cam0.png')
c1 = CudaComMat.CudaMat(img=img1)
c1.Upload()

firstSel = c1.CreateILBP(Flatten=True)


# time test
if TimeTest == True:
    for i in range(0, 100):
        start = time.perf_counter()
        firstSel = c1.CreateILBP(Flatten=False)
        end = time.perf_counter()
        elapsed = end - start
        print("elapsed time = {:.12f} seconds".format(elapsed))

shower1 = plt.imshow(c1.ownMat)

map1 = firstSel[:, :, 0]


# x,y
for i in range(0, c1.height()):
    for j in range(0, c1.width()):
        if 0 == map1[i, j]:
            continue
        if map1[i, j] == 1:
            plt.scatter(j, i, s=60, facecolors='none', edgecolors='r')
        else:
            row, col = CommonMath.GetArrow(i, j, map1[i, j]-2, 1)
            plt.annotate('', xy=(col, row), xycoords='data',
                xytext = (j, i), textcoords = 'data',
                arrowprops = dict(facecolor='green', arrowstyle='->', linewidth='1'))


plt.show()
