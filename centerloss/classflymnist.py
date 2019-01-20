"""mniist分类"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as pyp

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sample_path = r"/home/tensorflow01/oneday/mnist"

def mkdir(size):  #创建文件夹
    rootpath = os.path.join(sample_path, str(size))
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)
    return rootpath

if __name__ == '__main__':

    sess = tf.InteractiveSession()

    for i in range(10):
        count = 0
        print("i==>",i)
        rootpath = mkdir(i)
        lebal_file = open(rootpath + "/lebal.txt", "w")  # 创建标签文件
        for epoch in range(1000):
            print("epoch==>",epoch)
            imgdata,lebal = mnist.train.next_batch(1)
            # print(type(lebal))
            lebal = np.array(lebal,dtype=np.int)
            # print(type(lebal))
            # print("imgdata==>",imgdata[0])
            # print(lebal)
            # print("lebal==>",lebal[0])
            index = tf.argmax(lebal[0])
            index = sess.run(index)
            # print(index)
            if index == i:
                count +=1
                imgdata = np.array(np.reshape(imgdata[0]*255,[28,28]),dtype=np.uint8)
                # imgdata = imgdata[0].reshape([28,28],dty)
                # imgdata = imgdata*255
                # print(imgdata)
                # imgdata = np.uint8(imgdata)
                # print(imgdata)
                img = Image.fromarray(imgdata,"L")
                # img.save("/home/tensorflow01/oneday/mnist/0.jpg")
                img.save("/home/tensorflow01/oneday/mnist/{0}/{1}.jpg".format(i,count))
                lebal_file.write("{0}.jpg  {1}  {2}  {3}  {4}  {5}  {6}  {7}  {8}  {9}  {10}  {11}\n".format(
                    count,lebal[0][0],lebal[0][1],lebal[0][2],lebal[0][3],lebal[0][4],lebal[0][5],
                    lebal[0][6],lebal[0][7],lebal[0][8],lebal[0][9],i))
                # img.show()
            # break
        # break
