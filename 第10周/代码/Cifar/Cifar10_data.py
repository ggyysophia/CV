#该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow as tf
num_classes=10

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


#定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1                                            #如果是Cifar-100数据集，则此处为2
    result.height=32
    result.width=32
    result.depth=3                                           #因为是RGB三通道，所以深度是3

    image_bytes=result.height * result.width * result.depth  #图片样本总元素数量
    record_bytes=label_bytes + image_bytes                   #因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)  #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件  https://blog.csdn.net/fegang2002/article/details/83046584
    result.key,value=reader.read(file_queue)                 #使用该类的read()函数从文件队列里面读取文件

    record_bytes=tf.decode_raw(value,tf.uint8)               #读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    
    #因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    #剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    #这一步是将一维数据转换成3维数据
    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                           [result.depth,result.height,result.width])  

    #我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    #这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result                                 #返回值是已经把目标文件里面的信息都读取出来

def inputs(data_dir,batch_size,distorted):               #这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]   #拼接地址

    file_queue=tf.train.string_input_producer(filenames)     #根据已经有的文件地址创建一个文件队列
    read_input=read_cifar10(file_queue)                      #根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image=tf.cast(read_input.uint8image,tf.float32)   #将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch=num_examples_pre_epoch_for_train


    if distorted != None:                         #如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])          #首先将预处理好的图片进行剪切，使用tf.random_crop()函数

        flipped_image=tf.image.random_flip_left_right(cropped_image)    #将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)   #将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)    #将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数

        float_image=tf.image.per_image_standardization(adjusted_contrast)          #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24,24,3])                      #设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
                             #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train,tf.reshape(labels_train,[batch_size])

    else:                               #不对图像数据进行数据增强处理
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image=tf.image.per_image_standardization(resized_image)          #剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #这里使用batch()函数代替tf.train.shuffle_batch()函数  备注
        return images_test,tf.reshape(labels_test,[batch_size])

# tf.train.shuffle_batch
'''
tf.train.shuffle_batch
tf.train.shuffle_batch() 将队列中数据打乱后再读取出来．
函数是先将队列中数据打乱，然后再从队列里读取出来，因此队列中剩下的数据也是乱序的．

tensors：排列的张量或词典．
batch_size：从队列中提取新的批量大小．
capacity：队列中元素的最大数量．
min_after_dequeue：出队后队列中元素的最小数量，用于确保元素的混合级别．
num_threads：线程数量．
seed：队列内随机乱序的种子值．
enqueue_many：tensors中的张量是否都是一个例子．
shapes：每个示例的形状．(可选项）
allow_smaller_final_batch：为True时，若队列中没有足够的项目，则允许最终批次更小．(可选项）
shared_name：如果设置，则队列将在多个会话中以给定名称共享．(可选项）
name：操作的名称．(可选项）
————————————————
版权声明：本文为CSDN博主「阿卡蒂奥」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/akadiao/article/details/79645221
'''

# tf.train.string_input_producer官方说明
# 输出字符串到一个输入管道队列。
'''
参数解释
第一个参数·string_tensor·：1-D字符串Tensor。可以是一个文件名list；
第二个参数·num_epochs·: 可选参数，是一个整数值，代表迭代的次数，如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为num_epochs=N，生成器只能遍历tensor列表N次。
第三个参数shuffle： bool类型，设置是否打乱样本的顺序。一般情况下，如果shuffle=True，生成的样本顺序就被打乱了，在批处理的时候不需要再次打乱样本，使用 tf.train.batch函数就可以了;如果shuffle=False,就需要在批处理时候使用 tf.train.shuffle_batch函数打乱样本。
第四个参数seed: 可选的整数，是生成随机数的种子，在第三个参数设置为shuffle=True的情况下才有用。
第五个参数capacity：设置tensor列表的容量。
第六个参数shared_name：可选参数，如果设置一个‘shared_name’，则在不同的上下文环境（Session）中可以通过这个名字共享生成的tensor。
第七个参数name：可选，设置操作的名称。
第八个参数cancel_op：取消队列的操作（可选）。
————————————————
import tensorflow as tf

filename = ['../data/A.csv', '../data/B.csv', '../data/C.csv']

file_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=2)

reader = tf.WholeFileReader()
key, value = reader.read(file_queue)

with tf.Session() as session:
    session.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=session)
    for i in range(6):
        print(session.run([key, value]))


版权声明：本文为CSDN博主「李小白~」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40941722/article/details/104855857
'''