import CudaComMat
import cv2
import numpy
import matplotlib.pyplot as plt
import math
import CommonMath
import time


TimeTest = True

img1 = cv2.imread('D:\\test6.jpg', 0)
# c1 = CudaComMat.CudaMat(ReadPath='D:\\3d\\Calib0827\\0\\cam0.png')
c1 = CudaComMat.CudaMat(img=img1)
c1.Upload()

firstSel = c1.CreateILBPNet()

for i in range(0, c1.height()):
    for j in range(0, c1.width()):
        for r in range(16):
            if firstSel[i,j,r] == 14:
                firstSel[i,j,r] = 50
            '''if firstSel[i,j,r]<=10 or firstSel[i,j,r]>=20:
                firstSel[i,j,r] = firstSel[i,j,r]
            else:
                firstSel[i,j,r] = abs(firstSel[i,j,r]-14)'''


fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
ax = [fig.add_subplot(3,3,i+1) for i in range(9)]

c = 8

for i in range(3):
    for j in range(3):
        idx = i*3+j
        
        if idx ==4:
            continue
        ax[idx].axis('off')
        ax[idx].imshow(firstSel[:,:,c])
        c = c + 1
plt.subplot(3,3,5)
plt.imshow(img1)
plt.show()
plt.subplots_adjust( wspace=0)
plt.subplots_adjust( hspace=0)


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
