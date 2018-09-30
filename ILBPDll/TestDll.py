import ctypes


#load dll
dll = ctypes.cdll.LoadLibrary("ILBP.dll")

#call function
ret = dll.DllTest2()

print(ret)


#gen class
createInstance = dll.CreateInstance
createInstance.restype = ctypes.c_void_p
ILBP = createInstance()
classTest = dll.ClassTest
classTest.argtypes = [ctypes.c_void_p]
ret = classTest(ILBP)

print(ret)

destroyFunc = dll.DestroyInstance
destroyFunc.argtypes = [ctypes.c_void_p]
dll.DestroyInstance( ILBP )

