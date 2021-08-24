from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 用Image做图像RGB2GRAY
# 灰度化
path = '/Users/snszz/PycharmProjects/CV/第二周/代码/lenna.png'
# 1. Image
image = Image.open(path)  # 打开彩色图像 直接图像
# 转成数值 np.array(imagr)
image1 = image.convert('1')  # 把image转换成黑白  # 感觉噪声很大
# image1.save('black_white_Image_Lenna.jpg')

# 2.cv2
img = cv2.imread(path)
# imag.shape   512*512*3
type(img)  # numpy.ndarray
# 2.1 自己实现
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

cv2.imshow('image show gray', img_gray)
# cv2.imwrite('result1.jpg', img_gray)

# 2.2  cv2.cvtColor
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('./result2.jpg', gray)


# 3. plt  plt读入的数据是经过标准化的 是小数
img_plt = plt.imread(path)
plt.imshow(img_plt)
plt.title('lenna.png')

# img = cv2.imread("lenna.png", 1)   # falgs=1:三维数组， flags=0, 二维 黑白 而且不是原图 而且不是原图, 转成RGB和原图一样
# plt.imshow(img_plt)
# plt.title('lenna.png')
# 转成RGB
img = cv2.imread(path)  # BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
plt.imshow(img_rgb)   # 原图


# 3.1
img_plt = plt.imread(path)
img_plt_gray = rgb2gray(img_plt)   # rgb2gray 经过标准化

img = cv2.imread(path)  # 默认三维  B G R
img_gray1 = rgb2gray(img)    # 小数
plt.imshow(img_gray1)
plt.imshow(img_gray1)   # 两者的差异看不出来
plt.title('lenna.png')

# 4. matplotlib.image读取图片  与plt类似
# 利用matplotlib.image读取的图片，直接就生成了数组格式
import matplotlib.image as mpig
img_mpig = mpig.imread(path)#读取数据
print("img_mpig :",img_mpig .shape)
print("img_mpig :",type(img_mpig ))

# 5. skimage读取图片   R G B
'''scikit-image是基于scipy的一款图像处理包，
它将图片作为numpy数组进行处理，读取的数据正
好是numpy.ndarray格式。'''
import skimage.io as io
img_io = io.imread(path)#读取数据  R G B
print("img_io :",img_io .shape)
print("img_io :",type(img_io ))

# 6. keras读取图像
'''keras深度学习的框架，里面也是内置了读取图片的模块，
该模块读取的也不是数组格式，需要进行转换。'''
from keras.preprocessing.image import array_to_img, img_to_array, load_img

img_keras = load_img(path)#读取数据, 直接是图片

print("img_keras:",img_keras)
print("img_keras:",type(img_keras))
# 将图片转换为数组 使用keras里的img_to_array()
target = (512,512)
img_keras = img_keras.resize(target, Image.NEAREST)    # 还是图片

# keras中的img_to_array()函数并不等于nd.array().
# 主要区别在于，img_to_array()不管是2D shape还是3D shape image，
# 返回都是3D Numpy array. 而nd.array()本身不改变shape
# img_keras = img_to_array(img_keras)   # 报错
# print("img_keras:",img_keras.shape)
# print("img_keras:",type(img_keras))
# #可以使用使用np.array()进行转换
# mg_keras= np.array(img_keras)


'''
# 像素， 分辨率的单位，单位英寸对角线上像素的个数 单位PPI(Pixels Per Inch)
# 图像的基本单位，每个像素都有自己的颜色
# 性价比最高的算法， 能达成目的最简单的算法
'''

# 将灰度图转换为二值图，
# 首先需要将灰度图进行单位化， 因为plt.imread读取的图像是单位化之后的图像，直接基于该读取数据进行二值显示
img_plt = plt.imread(path)
img_plt_gray = rgb2gray(img_plt)
img_0_1 = np.where(img_plt_gray >= 0.5, 1, 0)
plt.imshow(img_0_1, cmap='gray')  # 缺少cmap='gray'， 得到的是彩色图

# 灰色图
plt.imshow(img_plt_gray, cmap = 'gray')

# 原图数组每个值除以255的图


# cv  的相关操作 ------------------------------------------
# 1. 读
img = cv2.imread(path)
# 2. 显示图片
cv2.imshow('show the image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#dv2.destroyWindow(wname)
# 说明1
'''
使用函数cv2.imshow(wname,img)显示图像，第一个参数是显示图像的窗口的名字，
第二个参数是要显示的图像（imread读入的图像），窗口大小自动调整为图片大小

cv2.waitKey  顾名思义等待键盘输入，单位为毫秒，即等待指定的毫秒数看是否有键盘输入，
若在等待时间内按下任意键则返回按键的ASCII码，程序继续运行。若没有按下任何键，超时后返回-1。
参数为0表示无限等待。不调用waitKey的话，窗口会一闪而逝，看不到显示的图片。

cv2.destroyAllWindow()销毁所有窗口
cv2.destroyWindow(wname)销毁指定窗口'''

# 3. 存

'''
使用函数cv2.imwrite(file，img，num)保存一个图像。第一个参数是要保存的文件名，
第二个参数是要保存的图像。可选的第三个参数，它针对特定的格式：对于JPEG，
其表示的是图像的质量，用0 - 100的整数表示，默认95;对于png ,第三个参数表示的是压缩级别。默认为3.
注意:
cv2.IMWRITE_JPEG_QUALITY类型为 long ,必须转换成 int
cv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小。

cv2.imwrite('1.png',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
cv2.imwrite('1.png',img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
'''
cv2.imwrite('1.png',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
cv2.imwrite('1.png',img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 4. 颜色转换
#彩色图像转为灰度图像
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#灰度图像转为彩色图像
img3 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# cv2.COLOR_X2Y，其中X,Y = RGB, BGR, GRAY, HSV, YCrCb, XYZ, Lab, Luv, HLS

#----------------------------------------------------------------------------
# 例子
# 读入一副图像，按’s’键保存后退出，其它任意键则直接退出不保存
import cv2
img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('1.png',img)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

# ----------------------------------------------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图6-1中的矩阵
img = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
], dtype=np.uint8)

# 用matplotlib存储
plt.imsave('img_pyplot.jpg', img)
# 用OpenCV存储
cv2.imwrite('img_cv2.jpg', img)
# ----------------------------------------------------------------
'''
6.2.2 基本图像处理
存取图像

读图像用cv2.imread()，可以按照不同模式读取，一般最常用到的是读取单通道灰度图，
或者直接默认读取多通道。存图像用cv2.imwrite()，注意存的时候是没有单通道这一说的，
根据保存文件名的后缀和当前的array维度，OpenCV自动判断存的通道，另外压缩格式还可
以指定存储质量，来看代码例子：
'''

# cv 基础学习链接：
# https://www.cnblogs.com/shizhengwen/p/8719062.html

# ----------------------------------------------------------------
'''
缩小：下采样 降采样
扩大图像，上采样 ：内插值  最邻近差值

'''
