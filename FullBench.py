import Bench


acc,loss = Bench.Test("Inceptionv3",BenchData="CIFAR-100")

print(acc,loss)