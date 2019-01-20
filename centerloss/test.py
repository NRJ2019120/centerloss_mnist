"""测试列表保存参数"""
import tensorflow as tf

class NET():

    def __init__(self):
        self.v1 = tf.Variable(tf.truncated_normal(shape=[3,3],stddev=10),name="v1")
        self.v2 = tf.Variable(tf.truncated_normal(shape=[3,3],stddev=10),name="v2")
        self.add()
    def add(self):
        self.sum = self.v1+self.v2

if __name__ == '__main__':

    net = NET()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver([net.v1])

    with tf.Session() as sess:

        saver.restore(sess, "./test_params.ckpt")
        print("Model restored.")
        # sess.run(init_op)
        # net.v2.initializer.run()
        sum = sess.run(net.sum)
        # saver.save(sess,"./test_params.ckpt")
        print(sum)
