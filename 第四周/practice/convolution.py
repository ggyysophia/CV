import cv2
import numpy as np


def convolution(gray, kernel, strides=1):
    kernel_size = kernel.shape[0]
    radius = kernel_size // 2
    gray_pad = np.pad(gray, ((radius, radius), (radius, radius)), 'constant')
    h = int((gray_pad.shape[0] - kernel_size) / strides + 1)
    w = int((gray_pad.shape[1] - kernel_size) / strides + 1)
    gray_new = np.zeros((h, w))
    # convolution
    for i in range(h):
        for j in range(w):
            gray_new[i, j] = np.sum(
                (gray_pad[i * strides:i * strides + kernel_size, j * strides:j * strides + kernel_size]) * kernel)
    return gray_new.astype(np.uint8)


def convolution1(gray, kernel1, kernel2, strides=1):
    kernel_size = kernel1.shape[0]
    radius = kernel_size // 2
    gray_pad = np.pad(gray, ((radius, radius), (radius, radius)), 'constant')
    h = int((gray_pad.shape[0] - kernel_size) / strides + 1)
    w = int((gray_pad.shape[1] - kernel_size) / strides + 1)
    gray_new = np.zeros((h, w))
    # convolution
    for i in range(h):
        for j in range(w):
            x = np.sum(
                (gray_pad[i * strides:i * strides + kernel_size, j * strides:j * strides + kernel_size]) * kernel1)
            y = np.sum(
                (gray_pad[i * strides:i * strides + kernel_size, j * strides:j * strides + kernel_size]) * kernel2)
            gray_new[i, j] = np.sqrt(x ** 2 + y ** 2)
    return gray_new.astype(np.uint8)

if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path, 1)
    # resize
    ratio = None
    if ratio:
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    filter1 = convolution(gray, kernel1, strides=1)
    cv2.imshow('lenna-filter1', filter1)
    cv2.waitKey(0)



    filter2 = convolution(gray, kernel2, strides=1)
    cv2.imshow('lenna-filter2', filter2)
    cv2.waitKey(0)

    filter3 = convolution1(gray, kernel1, kernel2, strides=1)
    cv2.imshow('lenna-filter', filter3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path, 1)
    # resize
    ratio = None
    if ratio:
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    filter1 = convolution(gray, kernel1, strides=1)
    cv2.imshow('lenna-filter1', filter1)
    cv2.waitKey(0)



    filter2 = convolution(gray, kernel2, strides=1)
    cv2.imshow('lenna-filter2', filter2)
    cv2.waitKey(0)

    filter3 = convolution1(gray, kernel1, kernel2, strides=1)
    cv2.imshow('lenna-filter', filter3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
