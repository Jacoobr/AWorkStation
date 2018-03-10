import cv2
import sys
img = cv2.imread('E:\zqt_picture\PT2_SER4_019.tif')
if(img == None):
    print('error')
    sys.exit(-1)
cv2.imshow('img',img)
cv2.waitKey(0)
