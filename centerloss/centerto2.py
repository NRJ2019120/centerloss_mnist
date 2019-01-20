import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

class NET():

    def __init__(self):
        self.x = tf.placeholder("float", [None, 28, 28, 1])  # 图片输入格式"NHWC"
        self.y_ = tf.placeholder("float", [None, 10])  # 图片标签输入
        self.conv1_1w = tf.Variable(tf.truncated_normal(shape=[3,3,1,64],stddev=0.01))
        self.conv1_2w = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=0.01))

        self.conv2_1w = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],stddev=0.01))
        self.conv2_2w = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=0.01))

        self.conv3_1w = tf.Variable(tf.truncated_normal(shape=[3,3,128,256],stddev=0.01))
        self.conv3_2w = tf.Variable(tf.truncated_normal(shape=[3,3,256,256],stddev=0.01))
        self.conv3_3w = tf.Variable(tf.truncated_normal(shape=[3,3,256,256],stddev=0.01))

        self.line1_w = tf.Variable(tf.truncated_normal(shape=[7*7*256,1024],stddev=0.01))
        self.line1_bias = tf.Variable(tf.zeros([1024]))

        self.line2_w = tf.Variable(tf.truncated_normal(shape=[1024,32],stddev=0.01))
        self.line2_bias = tf.Variable(tf.zeros([32]))

        self.line3_w = tf.Variable(tf.truncated_normal(shape=[32,2],stddev=0.01))
        self.line3_bias = tf.Variable(tf.zeros([2]))

        self.line4_w = tf.Variable(tf.truncated_normal(shape=[2,10],stddev=0.01))
        self.line4_bias = tf.Variable(tf.zeros([10]))

        self.forward()

    def forward(self):

        conv1_1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_1w, strides=[1, 1, 1, 1], padding="SAME") ) # 28*28*64
        conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, self.conv1_2w, strides=[1, 1, 1, 1], padding="SAME") ) # 28*28*64
        maxpool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")  # 14*14*64

        conv2_1 = tf.nn.relu(tf.nn.conv2d(maxpool_1, self.conv2_1w, strides=[1, 1, 1, 1], padding="SAME") )    # 14*14*128
        conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, self.conv2_2w, strides=[1, 1, 1, 1], padding="SAME")  )     # 14*14*128
        maxpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")         #7*7*128

        conv3_1 = tf.nn.relu(tf.nn.conv2d(maxpool_2, self.conv3_1w, strides=[1, 1, 1, 1], padding="SAME") )  #7*7*256
        conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, self.conv3_2w, strides=[1, 1, 1, 1], padding="SAME")  )   #7*7*256
        conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, self.conv3_3w, strides=[1, 1, 1, 1], padding="SAME")  )   #7*7*256

        line = tf.reshape(conv3_3,[-1, 7 * 7 * 256])


        fc_1 = tf.nn.relu_layer(line,self.line1_w,self.line1_bias)  #1024
        self.keep_drop = tf.placeholder(dtype=tf.float32)  # 纯量没有形状shape=[1],报错
        fc1_drop = tf.nn.dropout(fc_1, self.keep_drop)
        fc_2 = tf.matmul(fc1_drop,self.line2_w)+self.line2_bias #32
        self.fc_to2 = tf.matmul(fc_2,self.line3_w)+self.line3_bias  #2

        # mean,variance = tf.nn.moments(self.fc_to2,0)
        # self.batchnormal_out = tf.nn.batch_normalization(
            # x=self.fc_to2,mean=mean,variance=variance,offset=0,scale=200,variance_epsilon=0.01)
        # self.y_out = tf.matmul(self.fc_3,self.line4_w)+self.line4_bias  #10

if __name__ == '__main__':

    net = NET()
    # init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    # with tf.Session() as sess:
    # saver = tf.train.Saver()
    saver = tf.train.Saver({"conv1_1w": net.conv1_1w, "conv1_2w": net.conv1_2w, "conv2_1w": net.conv2_1w,
                            "conv2_2w": net.conv2_2w, "conv3_1w": net.conv3_1w, "conv3_2w": net.conv3_2w,
                            "conv3_3w": net.conv3_3w, "line1_w": net.line1_w, "line1_bias": net.line1_bias,
                            "line2_w": net.line2_w, "line2_bias": net.line2_bias, "line3_w": net.line3_w,
                            "line3_bias": net.line3_bias, "line4_w": net.line4_w, "line4_bias": net.line4_bias})
    # saver.restore(sess, "./params.ckpt")
    saver.restore(sess, "./params_center.ckpt")
    print("Model restored.")
    plt.ion()
    for epoch in range(1000):
        xs ,ys = mnist.train.next_batch(1)
        print("lebal==>", ys[0])
        index = tf.argmax(ys[0])
        index = sess.run(index)
        print("index==>",index)
        xs_ = np.reshape(xs,[-1,28,28,1])
        fc_to2 = sess.run(net.fc_to2,
                                 feed_dict={net.x:xs_, net.y_:ys,net.keep_drop: 1.0})
        print("fc_to==>",fc_to2)
        print(fc_to2[0][0])
        if index == 0:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="r")
        elif index == 1:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="c")
        elif index == 2:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="y")
        elif index == 3:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="b")
        elif index == 4:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="m")
        elif index == 5:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="g")
        elif index == 6:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="k")
        elif index == 7:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="orange")
        elif index == 8:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="indigo")
        else:
            plt.scatter(x=int(fc_to2[0][0]/10),y=int(fc_to2[0][1]/10),c="skyblue")
        plt.pause(0.1)
        #print("batchnormal_out==>",batchnormal_out)
        # break

    # plt.ioff()