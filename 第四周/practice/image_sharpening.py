import cv2
import numpy as np

# run contrl + shift + R
if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path, 1)
    print(img.shape, 'img.shape')
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape, 'gray.shape')
    # show
    cv2.imshow('lenna.gray', gray)
    cv2.waitKey(0)

    # sharpening
    kernel1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    new_gray = np.zeros(gray.shape)
    # padding
    gray_pad = np.pad(gray, ((1,1), (1,1)), 'constant')
    for i in range(new_gray.shape[0]):
        for j in range(new_gray.shape[1]):
            new_gray[i,j] = np.sum(gray_pad[i:i+3, j:j+3] * kernel1)
    cv2.imshow('lenna_sharpening1', new_gray.astype(np.uint8))
    cv2.waitKey(0)

    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_gray1 = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            new_gray1[i,j] = np.sum(gray_pad[i:i+3, j:j+3] * kernel2)
    cv2.imshow('lenna_sharpening2', new_gray1.astype(np.uint8))
    cv2.waitKey(0)

    kernel3 = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
    new_gray3 = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            new_gray3[i,j] = np.sum(gray_pad[i:i+3, j:j+3] * kernel3)
    cv2.imshow('lenna_sharpening2', new_gray3.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# sharpening :图像的锐化，---图像边缘的增强
# kernel2的效果好一些

