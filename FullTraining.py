


import Training
import Bench



testNet = []
testSet = []


testNet = "ILBPNet"

testSet.append("IIIT5K")


accData = dict()
lossData =dict()

for t in testSet:        
    Training.SetTrain(t,testNet,GlobalEpoche= 1,rdnSize=10000,Epoche=50,batchSize=64)
    acc,loss = Bench.Test(testNet,BenchData=t)
    accData[t] = acc
    lossData[t] = loss


print(accData)
print(lossData)