import matplotlib.pyplot as plt
import numpy as np

img = np.random.randn(100, 100)

plt.figure()
plt.imshow(img)
plt.arrow(10,10,20,30,head_length=5,color='r')



plt.scatter(25, 50, s=60,facecolors='none', edgecolors='r')

plt.show()