import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm

X = np.random.randint( 10,size=(10,20) )
y = np.zeros((10,1))
for i in range(10):
    X[i,:] += 100*i
    y[i] = i


clf = svm.SVC(verbose=1)
clf.fit(X.reshape(-1,20),y)


pre = clf.predict(X.reshape(-1,20))
print(pre)
