import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

import cv2


img = cv2.imread('D:\\TestImg\\test.jpg',0)
img = cv2.resize(img,(100,100))


fd, hog_image = hog(img, orientations=10, pixels_per_cell=(10, 10),
                    cells_per_block=(2, 2), visualize=True)


plt.imshow(hog_image)