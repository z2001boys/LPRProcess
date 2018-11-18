from ILBPLayer import MyLayer
from CNN_Module import ShuffleNetv2
def GetMdl( inputShape,ClassNum ):

    model = ShuffleNetv2(input_shape=inputShape, classes=ClassNum)

    return model