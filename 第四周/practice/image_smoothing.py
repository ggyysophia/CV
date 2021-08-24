import cv2
import numpy as np

# run contrl + shift + R
if __name__ == '__main__':
    # path = 'lenna.png'
    path = '身份证反.jpeg'
    img = cv2.imread(path, 1)
    print(img.shape, 'img.shape')
    # resize
    img = cv2.resize(img, (int(img.shape[1] * 0.15), int(img.shape[0] * 0.15)))
    print(img.shape, 'img.resize')
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape, 'gray.shape')
    # show
    cv2.imshow('lenna.gray', gray)
    cv2.waitKey(0)

    # smoothing1
    kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    new_gray = np.zeros(gray.shape)
    # padding
    gray_pad = np.pad(gray, ((1, 1), (1, 1)), 'constant')
    for i in range(new_gray.shape[0]):
        for j in range(new_gray.shape[1]):
            new_gray[i, j] = np.sum(gray_pad[i:i + 3, j:j + 3] * kernel1)
    cv2.imshow('lenna_smoothing1', new_gray.astype(np.uint8))
    cv2.waitKey(0)

    # smoothing2
    kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    new_gray1 = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            new_gray1[i, j] = np.sum(gray_pad[i:i + 3, j:j + 3] * kernel2)
    cv2.imshow('lenna_smoothing2', new_gray1.astype(np.uint8))
    cv2.waitKey(0)

    # 叠加
    # smoothing3

    new_gray2 = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            x = np.sum(gray_pad[i:i + 3, j:j + 3] * kernel1)
            y = np.sum(gray_pad[i:i + 3, j:j + 3] * kernel2)
            new_gray2[i, j] = np.sqrt(x ** 2 + y ** 2)
    cv2.imshow('lenna_smoothing12', new_gray2.astype(np.uint8))
    cv2.waitKey(0)

# 叠加的平滑比较好


'''
cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)

InputArray src	输入图片
OutputArray dst	输出图片
Size	输出图片尺寸
fx, fy	沿x轴，y轴的缩放系数
interpolation	插入方式
interpolation 选项所用的插值方法：

INTER_NEAREST

最近邻插值

INTER_LINEAR

双线性插值（默认设置）

INTER_AREA

使用像素区域关系进行重采样。

INTER_CUBIC

4x4像素邻域的双三次插值

INTER_LANCZOS4

8x8像素邻域的Lanczos插值
'''