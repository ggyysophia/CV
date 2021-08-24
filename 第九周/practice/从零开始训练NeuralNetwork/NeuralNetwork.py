import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set learning rate
        self.lr = learningrate

        # to initialize the weights
        # a simple network ---2 layers---
        # wih: input--->hidden
        # who: hidden-->output
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 激活函数  _ 是一个函数， 如果直接写成
        # self.activation_function = scipy.special.expit(x)----已经是一个数值， 以后怎么调用呢
        # 希望把它封装成一个函数的形式 lambda x:x---->将其装换为 scipy.special.expit(x)的形式
        # sigmoid -->scipy.special.expit
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # input--hidden
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 反向， 计算误差
        outputs_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, outputs_errors * (1 - outputs_errors))

        # 更新权重
        self.who += self.lr * np.dot(outputs_errors * final_outputs * (1 - final_outputs),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                     np.transpose(inputs))

    def query(self, inputs):
        # 对于推理函数
        # input--->hidden
        hidden_inputs = np.dot(self.wih, inputs)
        # hidden activation
        hidden_outputs = self.activation_function(hidden_inputs)
        # hidden--->outputs
        final_inputs = np.dot(self.who, hidden_outputs)
        # output activation
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


if __name__ == '__main__':
    # inputnodes, hiddennodes, outputnodes, learningrate = 3, 3, 3, 0.3
    # n = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
    # n.query(inputs=[1.0, 0.5, -1.5])
    print('aa')


