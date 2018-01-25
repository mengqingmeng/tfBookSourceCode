import cv2
import numpy as np
img = cv2.imread("/home/mqm/Workspace/DL/TensorFlow源代码/源代码/lena.jpg")
M_copy_img = np.array([
    [0, 0.8, 0],
    [1, 0, 0]
], dtype=np.float32)
img_change = cv2.warpAffine(img, M_copy_img,(512,512))
cv2.imshow("test",img_change)
cv2.waitKey(0)
