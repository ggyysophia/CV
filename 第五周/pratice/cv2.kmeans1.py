import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('lenna.png')
print(img.shape)
#(486, 654, 3)
#图像二维像素转换为一维
#转换成3列
data = img.reshape((-1,3))
data = np.float32(data)

#定义终止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置初始中心的选择
# flags = cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_PP_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

#图像转换回uint8二维类型
print(labels, 'labels')
centers = np.uint8(centers)
print(centers, 'centers')
res1 = centers[[0, 1, 2, 2, 3, 3]]
print(res1, 'res1')
# res = centers[labels.flatten()]
# print(res, 'centers flatten')
# # dst = res.reshape((img.shape))
# #
# # cv2.imwrite("img/autumn4.png",dst)

