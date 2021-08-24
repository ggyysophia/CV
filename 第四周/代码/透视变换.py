import cv2
import numpy as np

img = cv2.imread('photo.JPG')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
src = np.float32([[765, 1658], [932, 1656], [759, 2014], [983, 2026]])
dst = np.float32([[0, 0], [932, 1656], [765, 2014], [932, 2014]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (932,2014))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
