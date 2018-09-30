import cv2
import numpy


def simpleShow(img):
    cv2.imshow("test", img)
    cv2.waitKey()


def PutLabel(img, label, scale, thinckness, r, c):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    boxSize, baseLine = cv2.getTextSize(label, fontface, scale, thinckness)
    cv2.rectangle(img,  # image
                  (c, r),  # p1
                  (c+boxSize[0], r + boxSize[1]),  # p2
                  (0, 255, 0),  # color
                  cv2.FILLED)  # fill type
    cv2.putText(img, label,
                (c, r+baseLine*2),
                fontface, scale, (0, 0, 0))


def AddLabel(img, labelObj):
    imgToShow = numpy.zeros((img.shape[0], img.shape[1], 3), dtype=numpy.uint8)

    if len(img.shape) == 3:
        imgToShow = img
    else:
        imgToShow[:, :, 0] = img
        imgToShow[:, :, 1] = img
        imgToShow[:, :, 2] = img

    for o in labelObj:
        label = o.Label
        PutLabel(imgToShow, label, 0.6, 3, o.Box.r1, o.Box.c1)
        cv2.rectangle(imgToShow,  # image
                      (o.Box.c1, o.Box.r1),  # p1
                      (o.Box.c2, o.Box.r2),  # p2
                      (0, 255, 0))  # color
        simpleShow(imgToShow)
    return imgToShow
