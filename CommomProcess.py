import os


def ToLabelFile(path, target):
    # get all files' and folders' names in the current directory
    filenames = os.listdir(path)
    folders = []

    for filename in filenames:  # loop through all the files and folders
        # check whether the current object is a folder or not
        if os.path.isdir(os.path.join(os.path.abspath("."), filename)):
            folders.append(filename)

    with open(target, 'w') as file:
        for f in folders


ToLabelFile('D:\\LicensePlateDataSet\\', 'DatasetList\\LicesnePlate_test1.txt')
