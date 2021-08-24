import caffe2
import tensorflow as tf
'''
# 查看tensorflow 的版本 出现多个.py  run ---Edits, configurations
# print('version: ', tf.__version__)  # 1.14.0

# Tensorflow, Pytorch,
# 相关优化 BGD, SGD
# 拓展 Adam, RMSprop

# keras 框架 类似opencv, 是由纯python编写的基于tneano和tensorflow的深度学习框架
# softmax layers as the output layer， normalization to [0, 1]
# softmax 用在网络的最后一层， 分类、多分类 softmax in [0, 1] --概率---》从而实现多分类
# y_i = exp(v_i) / sum(exp(v_i)), 0 < y_i < 1, sum(y_i) = 1
# 分类结果， 就是softmax的概率大小的排序，
'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)


from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
# network.add(layers.Dense())

# [5] label的制作
'''
tf = 1.14  k
7 ---->[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
cat, dog, pig     cat----->[1, 0, 0]
[0.1, 0.1, 0.8]------>[0, 0, 1] --- pig
fit函数不区分网络结构， fit中的结构不会发生变化
网络结构会不同，  但是训练过程相同

'''

# network.fit(train_images, train_images, epochs=5, batch_size=128)

# [6]
'''
测试数据的输入， 检验网络学习
识别效果与硬件相关（CPU， GPU）
verbose=1：输出进度条进度
'''
# test_loss, test_acc = network.evaluate(test_images, test_iamges, test_labels, verbose=1)
# print("test_loss: ", test_loss)
# print("test_acc: ", test_acc)

# [7]
'''
输入一张手写数字图片到网络中， 看看它的识别效果
'''

# [8]
'''
完整数数据处理， 数据预处理（归一化， 滤波，直方图，均衡化…………………………），网络构建，
推理和训练
'''

