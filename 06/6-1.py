#coding=utf-8
import cv2
# img = cv2.imread("/home/mqm/Workspace/DL/TensorFlow源代码/源代码/lena.jpg")
# img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# turn_green_hsv = img_hsv.copy()
# turn_green_hsv[:,:,0] = (turn_green_hsv[:,:,0] - 30 ) % 180
# turn_green_img = cv2.cvtColor(turn_green_hsv,cv2.COLOR_HSV2BGR)
# cv2.imshow("test",turn_green_img)
# cv2.waitKey(0)
#读取图片
# img = cv2.imread("/home/mqm/Workspace/DL/TensorFlow源代码/源代码/lena.jpg")
# cv2.imshow("test",img)
# cv2.waitKey(0)
img =cv2.imread("/home/mqm/Workspace/DL/TensorFlow源代码/源代码/lena.jpg")
img_hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:,:,0] = (turn_green_hsv[:,:,0]-30) % 180
turn_green_img = cv2.cvtColor(turn_green_hsv,cv2.COLOR_HSV2BGR)
cv2.imshow("test",turn_green_img)
cv2.imshow("test2",img_hsv)
cv2.waitKey(0)