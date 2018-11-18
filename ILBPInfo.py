


from Models import MobileNetv2
from Models import ILBPNet


class ILBPInfo:
    def __init__(self):
        self.depth = 2
        
    def GetMdl(self,imgw,imgh,classNum):
        return MobileNetv2.GetMdl((imgw,imgh,self.depth),classNum)

    def GetName(self):
        return "ILBPNet"

    def GetFun(self):
        return 