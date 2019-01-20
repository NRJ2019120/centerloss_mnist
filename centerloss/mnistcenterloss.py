"""center中心点作为变量学习"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

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


        self.c0_w = tf.Variable(initial_value=[1000,1000],dtype=tf.float32)
        self.c1_w = tf.Variable(initial_value=[-1000,1000],dtype=tf.float32)
        self.c2_w = tf.Variable(initial_value=[-800,-800],dtype=tf.float32)
        self.c3_w = tf.Variable(initial_value=[-1000,0],dtype=tf.float32)
        self.c4_w = tf.Variable(initial_value=[1000,-800],dtype=tf.float32)
        self.c5_w = tf.Variable(initial_value=[800,0],dtype=tf.float32)
        self.c6_w = tf.Variable(initial_value=[500,1000],dtype=tf.float32)
        self.c7_w = tf.Variable(initial_value=[-300,-500],dtype=tf.float32)
        self.c8_w = tf.Variable(initial_value=[-300,500],dtype=tf.float32)
        self.c9_w = tf.Variable(initial_value=[300,-800],dtype=tf.float32)

        self.forward()
        self.backward()
        self.evolution()

    def forward(self):

        conv1_1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_1w, strides=[1, 1, 1, 1], padding="SAME")) # 28*28*64
        conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, self.conv1_2w, strides=[1, 1, 1, 1], padding="SAME")) # 28*28*64
        maxpool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")  # 14*14*64

        conv2_1 = tf.nn.relu(tf.nn.conv2d(maxpool_1, self.conv2_1w, strides=[1, 1, 1, 1], padding="SAME"))    # 14*14*128
        conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, self.conv2_2w, strides=[1, 1, 1, 1], padding="SAME"))     # 14*14*128
        maxpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")  #7*7*128

        conv3_1 = tf.nn.relu(tf.nn.conv2d(maxpool_2, self.conv3_1w, strides=[1, 1, 1, 1], padding="SAME"))  #7*7*256
        conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, self.conv3_2w, strides=[1, 1, 1, 1], padding="SAME"))   #7*7*256
        conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, self.conv3_3w, strides=[1, 1, 1, 1], padding="SAME"))   #7*7*256

        line = tf.reshape(conv3_3,[-1, 7 * 7 * 256])


        fc_1 = tf.nn.relu_layer(line,self.line1_w,self.line1_bias)  #1024
        self.keep_drop = tf.placeholder(dtype=tf.float32)  # 纯量没有形状shape=[1],报错
        fc1_drop = tf.nn.dropout(fc_1, self.keep_drop)
        fc_2 = tf.matmul(fc1_drop,self.line2_w)+self.line2_bias   #32
        self.fc_3 = tf.matmul(fc_2,self.line3_w)+self.line3_bias  #2

        self.y_out = tf.matmul(self.fc_3,self.line4_w)+self.line4_bias  #10

    def backward(self):
        # self.y_[:,0].reshape(1,-1)
        self.centerloss = (1/512)*tf.reduce_mean(
                                tf.matmul(tf.reshape(self.y_[:,0],[1,-1]),tf.square(self.fc_3 - self.c0_w)) + tf.matmul(tf.reshape(self.y_[:,1],[1,-1]),tf.square(self.fc_3 - self.c1_w)) +
                                tf.matmul(tf.reshape(self.y_[:,2],[1,-1]),tf.square(self.fc_3 - self.c2_w)) + tf.matmul(tf.reshape(self.y_[:,3],[1,-1]),tf.square(self.fc_3 - self.c3_w)) +
                                tf.matmul(tf.reshape(self.y_[:,4],[1,-1]),tf.square(self.fc_3 - self.c4_w)) + tf.matmul(tf.reshape(self.y_[:,5],[1,-1]),tf.square(self.fc_3 - self.c5_w)) +
                                tf.matmul(tf.reshape(self.y_[:,6],[1,-1]),tf.square(self.fc_3 - self.c6_w)) + tf.matmul(tf.reshape(self.y_[:,7],[1,-1]),tf.square(self.fc_3 - self.c7_w)) +
                                tf.matmul(tf.reshape(self.y_[:,8],[1,-1]),tf.square(self.fc_3 - self.c8_w)) + tf.matmul(tf.reshape(self.y_[:,9],[1,-1]),tf.square(self.fc_3 - self.c9_w)))

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_out, labels=self.y_))
        self.totalloss = self.cross_entropy + 0.2 * self.centerloss
        self.opt = tf.train.AdamOptimizer().minimize(self.totalloss)

    def evolution(self):
        self.correct_predictiion = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y_out, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictiion, dtype=tf.float32))

if __name__ == '__main__':

    net = NET()
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)
    # net.c0_w.initializer.run()
    # net.c1_w.initializer.run()
    # net.c2_w.initializer.run()
    # net.c3_w.initializer.run()
    # net.c4_w.initializer.run()
    # net.c5_w.initializer.run()
    # net.c6_w.initializer.run()
    # net.c7_w.initializer.run()
    # net.c8_w.initializer.run()
    # net.c9_w.initializer.run()

    saver_1 = tf.train.Saver({"conv1_1w": net.conv1_1w, "conv1_2w": net.conv1_2w, "conv2_1w": net.conv2_1w,
                            "conv2_2w": net.conv2_2w, "conv3_1w": net.conv3_1w, "conv3_2w": net.conv3_2w,
                            "conv3_3w": net.conv3_3w, "line1_w": net.line1_w, "line1_bias": net.line1_bias,
                            "line2_w": net.line2_w, "line2_bias": net.line2_bias, "line3_w": net.line3_w,
                            "line3_bias": net.line3_bias, "line4_w": net.line4_w, "line4_bias": net.line4_bias})
    saver_2 = tf.train.Saver()
    saver_2.restore(sess, "./center_params.ckpt")
    print("Model restored.")

    for epoch in range(1000000):
        xs ,ys = mnist.train.next_batch(512)
        xs_ = np.reshape(xs,[-1,28,28,1])
        # print(xs_)
        loss,_ = sess.run([net.totalloss,net.opt],
                             feed_dict={net.x : xs_,net.y_:ys,net.keep_drop:0.5})
        if epoch % 100 == 0:
            saver_1.save(sess,"./params_center.ckpt")
            saver_2.save(sess,"./center_params.ckpt")
            cross_entroty,loss,acc,fc_3 = sess.run([net.cross_entropy,net.totalloss,net.accuracy,net.fc_3],
                                     feed_dict={net.x: xs_, net.y_: ys, net.keep_drop: 1.0})
            print("totalloss=", loss, "cross_entroty",cross_entroty,"正确率=", acc)
            print("fc_3==>",fc_3[0])
            # break





