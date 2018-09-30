import ILBPClass
import cv2
import numpy

c = ILBPClass.ClassInstance()

print(c.ClassTest())

img = cv2.imread('D:\\test.jpg',0)
ret = c.ShowImage(img)
print(ret)