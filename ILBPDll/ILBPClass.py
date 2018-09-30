import ctypes


class ClassInstance:
    def __init__(self):
        self.dllHandler_ = ctypes.cdll.LoadLibrary('ILBP.dll')
        creatInstance = self.dllHandler_.CreateInstance
        creatInstance.restype = ctypes.c_void_p
        self.ILBP_=creatInstance()

    def __del__(self):
        destroyFunc = self.dllHandler_.DestroyInstance
        destroyFunc.argtypes = [ctypes.c_void_p]
        destroyFunc( self.ILBP_ )

    def ClassTest(self):
        Func = self.dllHandler_.ClassTest
        Func.argtypes = [ctypes.c_void_p]
        return Func( self.ILBP_ )

    def FastMapp(self):
        return self.dllHandler_.ClassTest(self.ILBP_)

    def ShowImage(self,img):
        Func = self.dllHandler_.ShowImgTest
        Func.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
        imgSize = img.shape
        return Func(img.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
            67,
            67)