from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

path = "D:\\Plates\\"
files = os.listdir(path)

globalCount = 0

for f in files:
    if os.path.isfile(path+f):
        img = cv2.imread(path+f)
        img = img.reshape((1,) + img.shape)
        singleCount = 0
        for b in datagen.flow(img, save_to_dir='D:\LicensePlateDataSet\\Plates', save_prefix='plate_'+str(globalCount), save_format='jpg'):
            singleCount = singleCount + 1
            if singleCount >= 1200:
                break

        globalCount = globalCount+1
