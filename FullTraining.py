'''

This is a project to have a quick view of machine learning
The project mainly used Keras, OpenCV, SKLearn, SKImage, and tensorflow.

Subprojects contain two main part: 
    1. MyKeras
    2. LabelImage
    3. Network Models : Inception, DarkNet, DenseNet, ShuffleNet, MobileNet, and ILBPNet

Mykeras contain a frist

The first authors of this project are "Egg" and "Blame", and the concat email is following.



'''


import Training
import Bench



testNet = []
testSet = []


testNet = "MobileNetv2"

testSet.append("IIIT5K")

accData = dict()
lossData =dict()

for t in testSet:        
    Training.SetTrain(t,testNet,GlobalEpoche= -1,rdnSize=10000,Epoche=200,batchSize=32,skLearn=True)
    acc,loss = Bench.Test(testNet,BenchData=t)
    accData[t] = acc
    lossData[t] = loss


print(accData)
print(lossData)