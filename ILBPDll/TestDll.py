import ctypes

dllPath = 'C:\\Users\\xinyo\\source\\repos\\CudaComMat\\x64\\Debug\\CudaMain.dll'
#load dll
dll = ctypes.cdll.LoadLibrary(dllPath)

#call function
ret = dll.DllTest()

print(ret)


#gen class
createInstance = dll.CreateComMat
createInstance.restype = ctypes.c_void_p
createInstance.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
ILBP = createInstance(1,20,20,0)
classTest = dll.CallTest
classTest.argtypes = [ctypes.c_void_p,ctypes.c_int]
ret = classTest(ILBP,10)

