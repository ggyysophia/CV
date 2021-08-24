import numpy as np
'''
    data_file = open('/Users/snszz/PycharmProjects/CV/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_test.csv')
    data_list = data_file.readlines()
    data_file.close()
    print('len(data_list): ', len(data_list))
    # print('data_list[0]: ',data_list[0])
    import matplotlib.pyplot as plt

    data_ = data_list[0]
    image = data_.strip().split(',')
    image = np.asfarray(image[1:]).reshape(28, 28)
    print(image.shape)
    image = image.astype(np.uint8)
    print(image.dtype)
    # plt.imshow(image, cmap='gray')
    # plt.show()
'''

'''
train_data_file = open('/Users/snszz/PycharmProjects/CV/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_train.csv')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    print('len(data_list): ', len(train_data_list))
    # print('data_list[0]: ',data_list[0])
    import matplotlib.pyplot as plt

    train_data_ = train_data_list[0]
    image = train_data_.strip().split(',')
    image = np.asfarray(image[1:]) / 255.0 * .099 + 0.01
    print('image', image.shape)
    # 设置图片与数值的对应关系
    targets = np.zeros(10) + 0.01
    targets[int(train_data_[0])] = 0.99
    print('train_data_[0]', train_data_[0])
    print('targets', targets)

    s1 = np.array(image, ndmin=2).T
    print('s1.shape', s1.shape)            # (784, 1)
    s2 = np.array(targets, ndmin=2).T   # ndmin z指定生成数组的最小维度   (10, 1)
    print('s2.shape', s2.shape)
'''

if __name__ == '__main__':
    #  https://blog.csdn.net/u013745804/article/details/79634196
    s3 = np.arange(1, 10)
    print(s3)
    s4 = s3[:, np.newaxis]    #
    print('s4', s4)
    print(s4.shape)           # (9, 1)
    s5 = s3[np.newaxis, :]
    print('s5', s5)
    print(s5.shape)          # (1, 9)

    # 使用 None 对array进行维度宽展
    y1 = s3[:, None]
    print(y1.shape, y1)    # (9, 1)
    y2 = s3[None, :]
    print(y2.shape, y2)     # (1, 9)

    # numpy.newaxis的本质就是None：
    #   如果硬要说numpy.newaxis有什么作用的话，其实就是让读代码的人更明确这是在创建一个新轴罢了。

# astype: float-->np.uint8  numpy数据类型转换需要调用方法astype()，
# 不能直接修改dtype。调用astype返回数据类型修改后的数据，但是源数据的类型不会变，
# 需要进一步对源数据的赋值操作才能改变