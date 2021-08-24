import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import tensorflow as tf


if __name__ == '__main__':
    flower = plt.imread('flower.png',1)
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    print(flower.shape)
    # cv2.imshow('flower', flower)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
    data = gray.reshape(1,332,444,1)
    print(data)
    # 浮雕效果 卷积核
    filter_ = np.array([[-1,-1,0],[-1,0,1],[0,1,1]]).reshape(3,3,1,1)
    # neural network
    # KNN ----> nearest neighbors
    # CNN ---->convolution neural network:卷积神经网络
    conv = tf.nn.conv2d(input = data,filter = filter_,strides=[1,1,1,1],padding='SAME')
    with tf.compat.v1.Session() as sess:
        image = sess.run(conv).reshape(332,444)
    plt.imshow(image,cmap = plt.cm.gray) #调整颜色

