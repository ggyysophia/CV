# 每一个接口在干嘛，
# 初始化， 输入层，输出层，中间层的节点个数
# 训练， 根据训练数据不断的更新权重值

# [1]
import numpy as np
import scipy.special


# 先搭框架，先写简单函数 再写复杂函数， 一步一运行，
class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化initialisation，设置输入层，中间层，和输出层的节点数。
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate  # 学习率

        # 初始化权重矩阵， 我们有两个权重矩阵， 一个是wih:输入--中间， 另一个 who:中间--输出
        # 初始化权重矩阵, 不会改变的都可以写在接口里 [-0.5, 0.5]
        # 需要初始化多少个权重矩阵和网络结构相关，
        # 初始化的范围可以自己设定，
        self.wih = np.random.rand(self.hnodes, self.innodes) - 0.5  # 输入和中间
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5  # 中间和输出

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, targets):
        # 自我训练过程分两步：
        # 1. 第一步是计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。
        # 2. 第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。
        # 隐藏层的输入
        hidden_inputs = np.dot(self.wih, inputs)  # w * h, self.wih是训练好的权重；
        # 激活作为输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层的激活作为输出
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)

        # 计算误差
        output_eerors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_eerors * final_outputs * (1 - final_outputs))
        self.who += self.lr * np.dot(output_eerors * final_outputs * (1 - final_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs  * (1 - hidden_outputs)), np.transpose(inputs))


    def query(self, inputs):
        # 推理函数， 没有反向
        # 根据输入计算出输出答案
        # 隐藏层的输入
        hidden_inputs = np.dot(self.wih, inputs)  # w * h, self.wih是训练好的权重；
        # 激活作为输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层的激活作为输出
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs




if __name__ == "__main__":
    inputs = [1, 2, 3]
    inputnodes, hiddennodes, outputnodes, learningrate = 3, 3, 3, 0.3
    network = NeuralNetWork(inputnodes, hiddennodes, outputnodes, learningrate)
    network.query(inputs)


'''
AI 算法的改动
scaled_input = 
'''