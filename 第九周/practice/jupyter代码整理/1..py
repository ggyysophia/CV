import numpy as np
from tensorflow import keras
import  tensorflow as tf

import matplotlib.pyplot as plt

# load  data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
print(test_images.shape)
print(test_labels)
print(np.unique(test_labels) ) #

# show
plt.imshow(test_images[0], cmap='gray')
plt.show()

# train， test---->验证数据准确度

# 构造神经元模型
# 3 layers--->10 类
import tensorflow as tf
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28, 28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
'''
# 查看参数和初始化信息
model.summary()

# 优化, loss function, metrics
train_images = train_images / 255
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)

# model.compile(optimizer=tf.optimizer.Adam(),
#              loss=tf.losses.sparse_categorical_crossentropy,
#              metrics=['accuracy'])

test_images = test_images / 255
model.evaluate(test_images, test_labels)

# 预测
c = model.predict([[test_images[0]]])
print(np.argmax(c))
print(test_labels[0])
plt.imshow(test_images[0])

# 1. 怎么查看TensorFlow的初始化权重值
tf.trainable_variables()

variable_name = [v.name for v in tf.trainable_variables()]
variable_name

# 2、获取权重：tf.get_default_graph().get_tensor_by_name('variable_name')
c = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
c

#------------------------------------------------------------------------------------------------

# 1. 防止过拟合
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('loss') < 0.4):
            print("\nLoss is low so cancelling training.")
            self.model.stop_training = True

callbacks =myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
training_images = training_images / 255.0
testing_images = testing_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

#------------------------------------------------------------------------------------------------

# 2.
# 前面是全连接网络
# 图片的旋转和平移
# 卷积神经网络
# CNN---max pooling:增强特征，减少数据
# 构建卷积神经网络
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
training_images = training_images / 255.0
testing_images = testing_images / 255.0

models = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                           input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

models.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
models.fit(training_images.reshape(-1, 28, 28, 1), training_labels, epochs=5)


#------------------------------------------------------------------------------------------------
# 3.modols.summary()
models.summary()

'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 26, 26, 64)        640       ---- 过滤器 3 * 3，+ bias = 1, (3 * 3 + 1) * 64 = 640, 26 = 28 - 3 + 1, 64 = 64个卷积核，64个图片
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 64)        0         ---- 13：使用MaxPooling2(2,2),尺寸减半，没有参数， 64个图片
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 11, 11, 64)        36928     ----11 = 13 - 3 + 1， 64 =  64个卷积核，  36928 = (3 * 3 * 64 + 1) * 64
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 5, 5, 64)          0         ---- 5：使用MaxPooling2(2,2),尺寸减半，没有参数， 64个图片
_________________________________________________________________
flatten_3 (Flatten)          (None, 1600)              0          --展平 5 * 5 * 64 = 1600
_________________________________________________________________
dense_6 (Dense)              (None, 128)               204928    --- 128 * (798 + 1)
_________________________________________________________________
dense_7 (Dense)              (None, 10)                1290      ----(128 + 1) * 10   # 全连接
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
'''
#------------------------------------------------------------------------------------------------
# 4.获取权重

import matplotlib.pyplot as plt

layer_output = [layer.output for layer in models.layers]
activation_model = tf.keras.models.Model(inputs = models.input,
                                         outputs = layer_output)
pred = activation_model.predict(testing_images[0].reshape(1,28, 28, 1))

len(pred)  # 7 个层
pred[0].shape   # (1, 26, 26, 64)
plt.imshow(pred[0][0, :, :, 1])    #一个模糊的鞋面
#------------------------------------------------------------------------------------------------
# 5. 项目案例




# -------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

model.fit_generator

