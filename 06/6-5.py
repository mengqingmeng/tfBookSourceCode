#coding=utf-8
import cv2
import random

img = cv2.imread("/home/mqm/Workspace/DL/TensorFlow源代码/源代码/lena.jpg")
width,height,depth = img.shape
#随机左上角的最大范围
img_width_box = width * 0.2
img_height_box = height * 0.2
for _ in range(9):
    start_pointX = int(random.uniform(0,img_width_box))
    start_pointY = int(random.uniform(0,img_height_box))
    copyImg = img[start_pointX:int(width*0.8), start_pointY:int(width*0.8)]
    cv2.imshow("test", copyImg)
    cv2.waitKey(0)
