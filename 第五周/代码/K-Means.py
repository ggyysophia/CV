# coding: utf-8

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('../pratice/lenna.png', 0)
print (img.shape)
# print(img, 'img')
# cv2.imshow('lenna', img)
# cv2.waitKey(0)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
print(dst, 'dst')
print(centers, 'centers')

# 标签与质心1
img_map = dst.copy()
for i in range(len(centers)):
   img_map[img_map == i] = centers[i]

# 标签与质心2
# centers = np.uint8(centers)
centers1 = centers[labels.flatten()]
centers1 = centers1.reshape(img.shape)
print(centers1, 'centers1')
#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
#显示图像
titles = ['原始图像', u'聚类图像', u'标签与质心1', u'标签与质心2']
images = [img, dst, img_map, centers1]
for i in range(4):
   plt.subplot(2, 2, 1 + i), plt.imshow(images[i], 'gray')   # plt.subplot(nrows, ncols, index, **kwargs)
   plt.title(titles[i])
   plt.ylabel(''), plt.xlabel('')
   plt.xticks([]),plt.yticks([])
plt.show()

'''
import numpy as np
c = [1, 2, 3, 2, 3]
d = [1.2, 3.4, 4.5]
dict1 = {i:j  for i,j in zip(set(c), d)}
print(dict1,'map')
np.random.seed(1111)
d = np.random.randint(1, 4, (4,5))
print(d, 'd')
d = d * 0.1 * 10
d[d==1] = 1.2
d[d==2] = 3.4
d[d==3] = 4.5
print(d, 'ddd')
'''