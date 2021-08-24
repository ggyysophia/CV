import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

def distance(X, Y, flags = 0):
    '''
    :param X: 数组1 n * 1
    :param Y: 数组2 n * m
    flags: = 0--欧式距离， 1 ----绝对值距离
    :return: 两者的欧式距离
    '''
    if flags == 0:
        dis = np.sqrt(np.sum((X - Y) ** 2, axis = 1))
    if flags == 1:
        dis = np.sum(np.abs(X - Y), axis = 1)
    return np.round(dis, 4)

def distance_detail(X, Y, flags = None):
    '''
    :param X:  n * 1
    :param Y: n * 1
    flags: = None--欧式距离， 1 ----绝对值距离
    :return: dis
    '''
    dis = 0
    if flags == 1:
        for i in range(X.shape[0]):
            dis += abs(X[i] - Y[i])
    else:
        for i in range(X.shape[0]):
            dis += (X[i] + Y[i]) ** 2
        dis = math.sqrt(dis)
    return np.round(dis, 4)

def intial_center(X, k, seed = 1234):
    '''
    :param X: 原始数据, 样本*特征
    :param k: 初始质心的个数
    :return: k个初始质心
    seed: 随机种子
    '''
    np.random.seed(seed)
    n = X.shape[0]
    random_k= np.random.choice(n, k)
    return X[random_k]

def main(X, k,  error, maxiter):
    # 1. 确定k，初始化质心
    init_center = intial_center(X, k)  # 初始化质心
    n1, n2 = X.shape   # n1:样本数, n2：特征数
    eps = np.inf      # 初始化误差
    count = 0         # 计数
    while eps > error:
        class_point = defaultdict(list)   # 存放每个点属于哪一类
        center = np.zeros((k, n2))    # 初始化更新后的质心
        # 2. 计算每个点到质心的距离， 再将点分配到该质心
        for i in range(n1):
            temp = distance(X[i], init_center)       # 计算出第i个点到每一个质心的距离
            index_min = np.argmin(temp)              # 找出距离最小的索引， 即为该点所属的类
            class_point[index_min].append([X[i], i])  # 将该点加入该类， 同时记录该点是第几个样本， 便于建立类别数据
        # 3. 更新质心
        for i in class_point:
            class_i_data = [k[0] for k in class_point[i]]    # 取出属于第i类的所有点
            center[i] = np.mean(np.array(class_i_data), axis = 0)  # 求mean为新的质心
        eps = np.sum((init_center - center) ** 2)      # 跟新两次质心之间的误差
        init_center = center                           # 更新初始质心
        count += 1                                     # 迭代次数+1
        # 4. 没有达到规定的误差，但是达到最大的迭代次数，也停止
        if count >= maxiter:
            break
        # 5. 把每个数据所属的类表示出来
        class_id = np.zeros(n1)
        for i in class_point:
            class_i_index = [j[1] for j in class_point[i]]   # 取出第i类所有的样本索引
            class_id[class_i_index] = i                      # 该索引下的类别为i
    return center, class_id.astype(np.int32)

if __name__ == '__main__':
    X = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
         ]
    X = np.array(X)
    center, class_id = main(X, k = 3, error = 1e-4, maxiter=500)
    print('质心:\n', center)
    print('每个点所属于的类别：\n', class_id)

    # 画图展示
    #用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    plt.figure(figsize=(4, 6))
    plt.subplot(2,1,1)
    plt.scatter(X[:, 0], X[:, 1], c = class_id, marker='*')
    #绘制标题
    plt.title("Kmeans-Basketball Data")
    #绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend()
    #设置右上角图例
    plt.tight_layout(pad=1.80)  #     调整每隔子图之间的距离
    plt.subplot(2,1,2)
    m = {0: '*', 1:'+', 2: 'D'}
    c = {0: 'r', 1:'g', 2: 'b'}

    for i in range(3):
        ind_i = np.where(class_id == i)
        print(ind_i, 'ind_i')
        x = X[ind_i][:, 0]
        y = X[ind_i][:, 1]
        plt.scatter(x, y, c=c[i], marker=m[i])
    plt.legend(["A","B","C"])
    # #绘制标题
    plt.title("Kmeans-Basketball Data_1")
    #绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute_1")
    plt.ylabel("points_per_minute_1")
    plt.show()











