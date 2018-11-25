import Bench


acc,loss = Bench.Test("Inceptionv3",BenchData="ICDAR03")

print(acc,loss)