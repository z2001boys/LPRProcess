
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
