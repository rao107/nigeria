import cv2 as cv
img = cv.imread('img/PXL_20230615_184150415.jpg')

cv.imshow('Display window', img)
k = cv.waitKey(0)
