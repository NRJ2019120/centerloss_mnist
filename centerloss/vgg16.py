import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import PIL

mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

class VGG16:
    def __init__(self):
        self.x = tf.placeholder([None,3,28,28],dtype=tf.float32)  #224*224*3
        self.y = tf.placeholder([None,10],dtype=tf.float32)
        self.conv1_1w =  tf.get_variable(tf.truncated_normal([3,3,3,64],dtype=tf.float32))
        self.conv1_2w =  tf.get_variable(tf.truncated_normal([3,3,64,64],dtype=tf.float32))

        self.conv2_1w =  tf.get_variable(tf.truncated_normal([3,3,64,128],dtype=tf.float32))
        self.conv2_2w =  tf.get_variable(tf.truncated_normal([3,3,128,128],dtype=tf.float32))

        self.conv3_1w =  tf.get_variable(tf.truncated_normal([3,3,128,256],dtype=tf.float32))
        self.conv3_2w =  tf.get_variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32))
        self.conv3_3w =  tf.get_variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32))

        self.conv4_1w =  tf.get_variable(tf.truncated_normal([3,3,256,512],dtype=tf.float32))
        self.conv4_2w =  tf.get_variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32))
        self.conv4_3w =  tf.get_variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32))

        self.conv5_1w =  tf.get_variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32))
        self.conv5_2w =  tf.get_variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32))
        self.conv5_3w =  tf.get_variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32))

        self.line_w1 = tf.get_variable(tf.truncated_normal([7*7*512,4096],dtype=tf.float32))
        self.line_bias = tf.get_variable(tf.zeros([4096],dtype=tf.float32))
        self.line_w2 = tf.get_variable(tf.truncated_normal([4096,4096],dtype=tf.float32))
        self.line_bias2 = tf.get_variable(tf.zeros([4096], dtype=tf.float32))
        self.line_w3 = tf.get_variable(tf.truncated_normal([4096,1000],dtype=tf.float32))
        self.line_bias3 = tf.get_variable(tf.zeros([1000], dtype=tf.float32))

        self.forward()
        self.backward()
        pass
    def forward(self):

        conv1_1 = tf.nn.conv2d(self.x, self.conv1_1w, strides=[1, 1, 1, 1], padding="SAME")  # 224*224*64
        conv1_2 = tf.nn.conv2d(conv1_1, self.conv1_2w, strides=[1, 1, 1, 1], padding="SAME")  # 224*224*64
        maxpool_1 = tf.nn.max_pool(conv1_2, ksize=[64, 2, 2, 64], strides=[1, 2, 2, 1], padding="VALID")  # 112*112*64

        conv2_1 = tf.nn.conv2d(maxpool_1, self.conv2_1w, strides=[1, 1, 1, 1], padding="SAME")  # 112*112*128
        conv2_2 = tf.nn.conv2d(conv2_1, self.conv2_2w, strides=[1, 1, 1, 1], padding="SAME")  # 112*112*128
        maxpool_2 = tf.nn.max_pool(conv2_2, ksize=[128, 2, 2, 128], strides=[1, 2, 2, 1], padding="VALID")  # 56*56*128

        conv3_1 = tf.nn.conv2d(maxpool_2, self.conv3_1w, strides=[1, 1, 1, 1], padding="SAME")  # 56*56*256
        conv3_2 = tf.nn.conv2d(conv3_1, self.conv3_2w, strides=[1, 1, 1, 1], padding="SAME")  # 56*56*256
        conv3_3 = tf.nn.conv2d(conv3_2, self.conv3_3w, strides=[1, 1, 1, 1], padding="SAME")  # 56*56*256
        maxpool_3 = tf.nn.max_pool(conv3_3, ksize=[256, 2, 2, 256], strides=[1, 2, 2, 1], padding="VALID")  # 28*28*256

        conv4_1 = tf.nn.conv2d(maxpool_3, self.conv4_1w, strides=[1, 1, 1, 1], padding="SAME")  # 28*28*512
        conv4_2 = tf.nn.conv2d(conv4_1, self.conv4_2w, strides=[1, 1, 1, 1], padding="SAME")  # 28*28*512
        conv4_3 = tf.nn.conv2d(conv4_2, self.conv4_3w, strides=[1, 1, 1, 1], padding="SAME")  # 28*28*512
        maxpool_4 = tf.nn.max_pool(conv4_3, ksize=[512, 2, 2, 512], strides=[1, 2, 2, 1], padding="VALID")  # 14*14*512

        conv5_1 = tf.nn.conv2d(maxpool_4, self.conv5_1w, strides=[1, 1, 1, 1], padding="SAME")  # 14*14*512
        conv5_2 = tf.nn.conv2d(conv4_1, self.conv5_2w, strides=[1, 1, 1, 1], padding="SAME")  # 14*14*512
        conv5_3 = tf.nn.conv2d(conv4_2, self.conv5_3w, strides=[1, 1, 1, 1], padding="SAME")  # 14*14*512
        maxpool_5 = tf.nn.max_pool(conv5_3, ksize=[512, 2, 2, 512], strides=[1, 2, 2, 1], padding="VALID")  # 7*7*512

        line = maxpool_5.reshape([-1,7*7*512])
        full_layer_1 = tf.nn.relu_layer(line,self.line_w1,self.line_bias)                         #4096
        full_layer_2 = tf.nn.relu_layer(full_layer_1,self.line_w2,self.line_bias2)                #4096
        full_layer_3 = tf.nn.relu_layer(full_layer_2,self.line_w3,self.line_bias3)                #1000

        softmax = tf.nn.softmax(full_layer_3)

    def backward(self):
        pass



if __name__ == '__main__':
    net = VGG16()
    init = tf.global_variables_initializer()

    for i in range(1000):
        with tf.Session() as sess:
            sess.run(init)
             _,loss = sess.run(net.opt,net.loss)

# Author:ZhengzhengLiu

import tensorflow as tf


# VGG_16全部使用3*3卷积核和2*2的池化核

# 创建卷积层函数
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    """
    param :
    input_op -- 输入tensor
    name -- 该层的名称
    kh -- 卷积核的高
    kw -- 卷积核的宽
    n_out -- 卷积核数目/输出通道数
    dh -- 步长的高
    dw -- 步长的宽
    p -- 参数（字典类型）
    return:
    A -- 卷积层的输出
    """
    n_in = input_op.get_shape()[-1].value  # 输入的通道数

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name=scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_con2d())
        biases = tf.get_variable(name=scope + "b", shape=[n_out], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0), trainable=True)
        conv = tf.nn.conv2d(input=input_op, filter=weights, strides=[1, dh, dw, 1], padding="SAME")
        Z = tf.nn.bias_add(conv, biases)
        A = tf.nn.relu(Z, name=scope)

        p[name + "w"] = weights
        p[name + "b"] = biases

        return A


# 创建最大池化层的函数
def maxpool_op(input_op, name, kh, kw, dh, dw):
    """
    param :
    input_op -- 输入tensor
    name -- 该层的名称
    kh -- 池化核的高
    kw -- 池化核的宽
    dh -- 步长的高
    dw -- 步长的宽
    return:
    pool -- 该层的池化的object
    """
    pool = tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding="SAME", name=name)
    return pool


# 创建全连接层的函数
def fc_op(input_op, name, n_out, p):
    """
    param :
    input_op -- 输入tensor
    name -- 该层的名称
    n_out -- 输出通道数
    p -- 参数字典
    return:
    A -- 全连接层最后的输出
    """
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name=scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        # biases不再初始化为0，赋予一个较小的值，以避免dead neuron
        biases = tf.get_variable(name=scope + "b", shape=[n_out], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # tf.nn.relu_layer对输入变量input_op与weights做矩阵乘法加上biases，再做非线性relu变换
        A = tf.nn.relu_layer(input_op, weights, biases, name=scope)

        p[name + "w"] = weights
        p[name + "b"] = biases

        return A


# 构建VGG_16网络的框架
def VGG_16(input_op, keep_prob):
    """
    param :
    input_op -- 输入tensor
    keep_prob -- 控制dropout比率的占位符
    return:
    fc8 -- 最后一层全连接层
    softmax -- softmax分类
    prediction --  预测
    p -- 参数字典
    """
    p = {}  # 初始化参数字典

    # 第一段卷积网络——两个卷积层和一个最大池化层
    # 两个卷积层的卷积核大小为3*3，卷积核数量均为64，步长s=1，输出均为：224*224*64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # 最大池化层采用的尺寸大小为：2*2，步长s=2，输出为：112*112*64
    pool1 = maxpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 第二段卷积网络——两个卷积层和一个最大池化层
    # 两个卷积层的卷积核大小为3*3，卷积核数量均为128，步长s=1，输出均为：112*112*128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    # 最大池化层采用的尺寸大小为：2*2，步长s=2，输出为：56*56*128
    pool2 = maxpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 第三段卷积网络——三个卷积层和一个最大池化层
    # 三个卷积层的卷积核大小为3*3，卷积核数量均为256，步长s=1，输出均为：56*56*256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 最大池化层采用的尺寸大小为：2*2，步长s=2，输出为：28*28*256
    pool3 = maxpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 第四段卷积网络——三个卷积层和一个最大池化层
    # 三个卷积层的卷积核大小为3*3，卷积核数量均为512，步长s=1，输出均为：28*28*512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 最大池化层采用的尺寸大小为：2*2，步长s=2，输出为：14*14*512
    pool4 = maxpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 第五段卷积网络——三个卷积层和一个最大池化层
    # 三个卷积层的卷积核大小为3*3，卷积核数量均为512，步长s=1，输出均为：14*14*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 最大池化层采用的尺寸大小为：2*2，步长s=2，输出为：7*7*512
    pool5 = maxpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    # 第六、七段 —— 含4096个隐藏节点的全连接层及dropout
    pool5_shape = pool5.get_shape().as_list()
    flattened_shape = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
    dense = tf.reshape(pool5, shape=[-1, flattened_shape], name="dense")  # 向量化

    fc6 = fc_op(dense, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name="fc7_drop")

    # 最后一层输出层含1000个节点,进行softmax分类
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax, 1)

    return prediction, softmax, fc8, psamsan


