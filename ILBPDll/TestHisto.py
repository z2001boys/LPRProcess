import CudaComMat
import numpy
import cv2 


import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


img = cv2.imread("D:\\TestImg\\test.jpg",0)


CudaComMat.buildHist(img)