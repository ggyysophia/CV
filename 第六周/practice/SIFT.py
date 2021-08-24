import numpy as np
import cv2
import matplotlib.pyplot as plt


# 1.  构建尺度空间
def getDoG(img, n, sigma0, S=None, O=None):
    """
    :param img: 输入图片
    :param n: 有几层用于特征提取   差分金字塔的数量
    :param sigma0: 输入的sigma
    :param S: 金字塔每层有几张gauss滤波后的图像
    :param O: 金字塔有几层
    :return: 返回差分金字塔和高斯金字塔
    """
    if S == None:
        S = n + 3  # 至少有4张(第一张和最后一张高斯金字塔无法进行特征提取，差分之后的第一张和最后一张也无法提取特征)
    if O == None:
        O = int(np.log(min(img.shape[0], img.shape[1]))) - 3  # 计算最大可以计算多少层
    k = 2 ** (1.0 / n)
    sigma = []
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)]
             for o in range(O)]     # 备注1   每一层 sigma 按照 k^1/s * sigma0 排列 下一层的sigma都要比上一层sigma的两倍
    sample = [undersampling(img, 1 << 0)  for o in range(O)]

    Guass_Pyramid = []
    for i in range(O):
        Guass_Pyramid.append([])     # 声明二维空list
        for j in range(S):     # 上网查找相关信息   高斯核大小随着sigma变化的效果更好  高斯核纬度固定的滤波效果不佳， 图像的分辨率越低， 所以高斯核大小应该随着sigma(尺度)变化而变化
            dim = int(6 * sigma[i][j] + 1)
            # dim = int(9)
            if dim % 2 == 0:
                dim += 1
            Guass_Pyramid[-1].append(
                    convolve(GuassianKernel(sigma[i][j], dim), sample[i], [dim // 2, dim // 2, dim // 2, dim // 2],
                             [1,1]))    # 在第i层添加第j张经过高斯卷积的 该图片四周扩展 5 // 2 =2 用于高斯卷积     x方向和y方向都移动一步，根据高斯核大小向图片填充0以计算边界点的卷积，
    DoG_Pyramid = [[Guass_Pyramid[o][s + 1] - Guass_Pyramid[o][s] for s in range(S)] for o in range(O)]   # 高斯金字塔每一层中上一张减去下一张得到高斯差分金字塔
    return DoG_Pyramid, Guass_Pyramid, O    # 返回高斯金字塔和高斯差分金字塔

def GuassianKernel(sigma, dim):
    '''
    :param sigma: 标准差
    :param dim: 高斯的维度(必须是奇数)
    :return: 高斯核
    '''
    temp = [t - (dim // 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    return result

def convolve(kernel, img, padding, strides):
    """
    :param kernel: 输入的核函数
    :param img: 输入的图片
    :param padding: 需要填充的位置
    :param strides: 高斯核移动的步长
    :return: 返回卷积的结果
    """
    result = None
    kernel_size = kernel.shape
    channel = []
    img_size = img.shape
    pad_img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3])),
                     'constant')  # pad是填充函数， 边界处卷积需要对边界外根据高斯核大小填0
    for j in range(0, img_size[0], strides[1]):   # 第j列strides是步长 本例步长为1 相当于遍历
        channel.append([])
        for k in range(0, img_size[1], strides[0]):  # 第i行
            val = (kernel * pad_img[j * strides[1] :j * strides[1] + kernel_size[0],
                                    k * strides[0]:k * strides[0] + kernel_size[1]]).sum()   # 卷积的定义， 相当于高斯核做加权和
            channel[-1].append(val)
    result = np.array(channel)
    return  result

def undersampling(img, step = 2):
    '''
    :param img: imput picture
    :param step: 降采样步长 默认为2（缩小2倍）
    :return: 返回降采样后结果
    '''
    return img[::step, ::step]
'''
showDoGimgs = 1
octaves = 3
n = 3
if showDoGimgs:
    plt.figure(1)
    for i in range(octaves):
        for j in range(n + 3):
            array = np.array(GuassianKernel([i][j], dtype = np.float32))
            plt.subpot(octaves, n + 3, j + (i) * octaves + 1)
            plt.imshow(array.astype(np.unit8), cmap='gray')
            plt.axis('off')
    plt.show()
    
    plt.figure(2)
    for i in range(octaves):
        for j in range(n + 2):
            array(DoG[i][j], dtype = np.float32)
            plt.subpot(octaves, n + 3, j + (i) * octaves + 1)
            plt.imshow(array.astype(np.unit8), cmap='gray')
            plt.axis('off')
plt.show()
            
'''

# 1.确定关键点
# 阈值化
# 在高斯差分金字塔中找极值点
# 调整极值点的位置
# 舍去低对比度的点
# 去除边缘效应

# 计算关键点的主方向

# 确定关键点的位置
# 第一步：阈值化：过小的点易受噪声干扰而变得不稳定， 所以将小于某个经验值（contrastThreshold=0.04）的极值点删除
# threshold = 0.5 * contrastThreshold / (n * 255 * SIFT_FIXPT_SCALE)
# np.abs(val) > threshold

# 第二步
'''
每一个采样点要和所有相邻的领域点比较， 即与同尺度的
8个相邻点和上下相邻尺度对应的9*2个点共26个点比较。
如果该点在DoG尺度空间本层以及上下两层的26个领域中是
最大的或者最小值时，则认为该点是图像在该尺度下的一个
极值点。
'''
# p14




























































































































































# 备注1
'''
   sigma1 = []
   for o in range(O):
       temp = []
       for s in range(S):
           temp.append((k ** s) * sigma0 * (1 << o))
       sigma1.append(temp)
   
   
   '''

# 备注2  np.convolve
"""
'''
[3,4]
[4]
[1,1,5,5]
[3,4]
[1,1,5,5]
  [3,4]
[1,1,5,5]
    [3,4]
[1,1,5,5]

       [3,4]
[1,1,5,5]

'''
# np.convolve(a, v, mode='full')
np.convolve([4,3], [1,1,5,5], mode='same')   # 前面的未全重叠 + 全重叠
np.convolve([4,3], [1,1,5,5], mode='valid')   # 全重叠
np.convolve([4,3], [1,1,5,5], mode='full')  # 前面的未全重叠 + 全重叠 + 后面的未全重叠
"""