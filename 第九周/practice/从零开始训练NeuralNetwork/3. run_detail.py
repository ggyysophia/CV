import matplotlib.pyplot as plt
import numpy as np
from  NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    inputnodes = 784
    hiddennodes = 100
    outputnodes = 10
    learningrate = 0.3
    epochs = 10
    n = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
    train_data_file = open('/Users/snszz/PycharmProjects/CV/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_train.csv')
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    print('len(data_list): ', len(train_data_list))
    # print('data_list[0]: ',data_list[0])
    # 10 轮数据的训练
    for e in range(epochs):
        for record in train_data_list:
            all_values = record.strip().split(',')
            inputs = (np.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
            # 图片标签与输出数组的对应
            targets = np.zeros(outputnodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # test

    score = []
    test_data_file = open('/Users/snszz/PycharmProjects/CV/第九周/代码/NeuralNetWork_从零开始/dataset/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    for record in test_data_list:
        all_values = record.strip().split(',')
        correct_number = int(all_values[0])
        inputs = (np.asfarray(all_values[1:])) / 255 * 0.99 + 0.01
        # 图片标签与输出数组的对应
        outputs = n.query(inputs)
        #找到数值最大的神经元对应的编号
        label = np.argmax(outputs)
        print("output reslut is : ", label)
        #print("网络认为图片的数字是：", label)
        if label == correct_number:
            score.append(1)
        else:
            score.append(0)

    print('score', score)

    # 计算图片判断的成功率
    score_accuracy = np.asarray(score)
    print("perfermance = ", score_accuracy.sum() / score_accuracy.size)







