
import string

# BuildStringClass``


def GetLicenseLabel():
    SortedClass = string.ascii_uppercase[:26]
    SortedClass = string.digits+SortedClass
    SortedClass = list(SortedClass)
    removeCase = ['O', 'I']
    for r in removeCase:
        SortedClass.remove(r)
    return SortedClass

def GetAllLabel():
    SortedClass = string.ascii_uppercase[:26]
    SortedClass = string.ascii_lowercase[:26]+SortedClass
    SortedClass = string.digits+SortedClass
    SortedClass = list(SortedClass)    
    return SortedClass

def DigitalOnly():
    SortedClass = string.digits
    SortedClass = list(SortedClass)    
    return SortedClass

def GetInt(r):
    SortedClass = []
    for i in range(r):
        SortedClass.append(str(i))
    return SortedClass
