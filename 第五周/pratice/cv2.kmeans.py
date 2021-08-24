# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)
print(img.shape, 'gray shape')
# cv2.imshow('gray lenna', img)
# cv2.waitKey(0)
#获取图像高度、宽度
rows, cols = img.shape
#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_MAX_ITER +
            cv2.TERM_CRITERIA_EPS, 10 , 1)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
print(compactness, 'compactness')
#生成最终图像
dst = labels.reshape(img.shape)
print(dst, 'dst')

# 质心与标签
centers = np.uint8(centers)
centers = centers[labels.flatten()]
centers = centers.reshape(img.shape)
print(centers, 'centers')

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
#显示图像
titles = ['原始图像', u'聚类图像', u'标签与质心1']
image = [img, dst, centers]
for i in range(3):
    plt.subplot(2,2, 1+i)
    plt.imshow(image[i], 'gray')
    plt.title(titles[i])
plt.show()







