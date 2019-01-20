import tensorflow as tf
import numpy as np

a = tf.constant([[8,9],[8,10]],dtype=tf.float32)
b = tf.constant([[4,6],[4,8]],dtype=tf.float32)
x = tf.constant(np.arange(100),shape=[10,10],dtype=tf.float32)
x_= x[:,0]

c = a-b
d= tf.square(c)
f= tf.reduce_mean(d)
# out =x[:,0]*d
if __name__ == '__main__':
    with tf.Session() as sess:
       c = sess.run(c)
       d = sess.run(d)
       f = sess.run(f)
       x = sess.run(x)
       x_ =sess.run(x_)
       # out =sess.run(out)

       print(c)
       print(d)
       print(f)
       print(x)
       print(x_)
       # print(out)
