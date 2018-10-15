from ILBPLayer import MyLayer
from CNN_Module import ShuffleNet
def GetMdl( ClassNum ):

    model = ShuffleNet(input_shape=(100, 100, 2), classes=ClassNum)

    return model