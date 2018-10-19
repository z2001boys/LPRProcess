import CudaComMat
import cv2
import numpy

img1 = cv2.imread('D:\\test.jpg',0)
img2 = cv2.imread('D:\\test.jpg',0)
#cv2.imshow("test",img)
#cv2.waitKey()

c1 = CudaComMat.CudaMat()
c1.AssignImgData(img1)
c1.Upload()
c1.Download()

print(c1.width())
print(c1.height())
print(c1.volumn())


cv2.imshow("test",img1)
cv2.waitKey()
