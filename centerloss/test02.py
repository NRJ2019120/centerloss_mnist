import matplotlib.pyplot as plt
import numpy as np

#
# plt.ion()
# for i in range(10):
#
#     plt.scatter(1,2+i,c="r")
#     # plt.clf()
#     plt.pause(0.5)

plt.ioff()

x = np.array(np.arange(1,101)).reshape((10,10))
print(x)
# print(x.T)
y =x[:][0]
print(y)
y = y.reshape([1,-1])
print(y)
print(y.T)
print(y*x)
print(y.T*x)