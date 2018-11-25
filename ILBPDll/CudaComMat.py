import ctypes
import numpy
import cv2


dllPath = 'D:\\HyperInspection\\CudaMain.dll'
#load dll
CudaComHandle = None
    

class CudaMat:
    '''def __init__(self):
        CudaComHandle = ctypes.cdll.LoadLibrary(dllPath)
        creatInstance = CudaComHandle.CreateComMatEmpty
        creatInstance.restype = ctypes.c_void_p
        self.ILBP_=creatInstance()'''

    def __init__(self,volumn=0,height=0,width=0,type=0,ReadPath = "",img=[]):
        
        global CudaComHandle
        if CudaComHandle==None:
            CudaComHandle = ctypes.cdll.LoadLibrary(dllPath)

        self.OnDevice = False
                
        creatInstance = CudaComHandle.CreateComMatEmpty
        creatInstance.restype = ctypes.c_void_p
        self.ILBP_=creatInstance()
        
        if ReadPath == "":
            if img != []:
                self.ownMat = img
            else:
                if(type==0):
                    self.ownMat = numpy.ndarray((volumn,height,width),dtype=numpy.uint8)
                else:
                    self.ownMat = numpy.ndarray((volumn,height,width),dtype=numpy.float)
        else:
            self.ownMat = cv2.imread(ReadPath,-1)

        self.AssignImgData(self.ownMat)

    def CreateILBPNet(self):
        if self.OnDevice == False:
            self.Upload()
        
        sp = self.ownMat.shape

        fullData = numpy.ndarray((sp[0],sp[1],16),dtype=numpy.uint8)

        func = CudaComHandle.CreateILBPNet
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]    
        func(self.ILBP_,
                fullData.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)))
        
        return fullData
                
                
    def CreateILBP(self,Flatten = False):
        if self.OnDevice == False:
            self.Upload()
                    
        sp = self.ownMat.shape

        if Flatten == False:
            firstSelectMap = numpy.ndarray((sp[0],sp[1],8),dtype=numpy.uint8)
            maxSelectMap = numpy.ndarray((sp[0],sp[1],8),dtype=numpy.uint8)

            func = CudaComHandle.CreateILBP
            func.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
            func(self.ILBP_,
                firstSelectMap.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
                maxSelectMap.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)))

            return firstSelectMap,maxSelectMap
        elif Flatten == True:
            FeatureMap = numpy.ndarray((sp[0],sp[1],3),dtype=numpy.uint8)
            func = CudaComHandle.CreateILBPFlatten
            func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
            func(self.ILBP_,
                FeatureMap.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)))

            return FeatureMap



    def __del__(self):
        destroyFunc = CudaComHandle.FreeAll
        destroyFunc.argtypes = [ctypes.c_void_p]
        destroyFunc( self.ILBP_ )

    def width(self):
        func = CudaComHandle.Width
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.c_void_p]
        ret = func(self.ILBP_)
        return ret

    def height(self):
        func = CudaComHandle.Height
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.c_void_p]
        ret = func(self.ILBP_)
        return ret


    def volumn(self):
        func = CudaComHandle.Volumn
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.c_void_p]
        ret = func(self.ILBP_)
        return ret
    
    def AssignImgData(self,img):
        func = CudaComHandle.AssignExtData
        sp = img.shape
        func.argtypes = [ctypes.c_void_p,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
            ctypes.c_void_p]
        imgPtr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        if len(sp)==2:
            func(self.ILBP_,1,sp[0],sp[1],0,imgPtr)
        else:
            func(self.ILBP_,sp[2],sp[0],sp[1],0,imgPtr)

    def GetMemAddr(self):
        func = CudaComHandle.GetMemAddr
        func.argtypes = [ctypes.c_void_p]
        ret = func(self.ILBP_)
        return ret
    
    def Upload(self):
        func = CudaComHandle.Upload
        func.argtypes = [ctypes.c_void_p]
        func(self.ILBP_)
        self.OnDevice = True
        


    def Sync(self):
        func = CudaComHandle.Upload
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        return func(self.ILBP_)

    def AllocOnGpu(self):
        func = CudaComHandle.GpuAlloc
        func.argtypes = [ctypes.c_void_p]
        func(self.ILBP_)

    def Download(self):
        func = CudaComHandle.Download
        func.argtypes = [ctypes.c_void_p]
        func(self.ILBP_)

    def Add(self,tobeAdd):
        func = CudaComHandle.Add
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func(self.ILBP_,tobeAdd.ILBP_)

    def TestSameImg(self):
        func = CudaComHandle.ProcImage
        func.argtypes = [ctypes.c_void_p]
        func(self.ILBP_)
    
    def CallTest(self):
        func = CudaComHandle.CallTest
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.c_void_p]
        ret = func(self.ILBP_,4)
        return ret
    