


import Training
import Bench

testNet = "ILBPNet"
testSet = "SHVT"


Training.SetTrain(testSet,testNet,GlobalEpoche= 5,rdnSize=3000,Epoche=6,batchSize=128)


acc,loss = Bench.Test(testNet,BenchData=testSet)

print(acc,loss)