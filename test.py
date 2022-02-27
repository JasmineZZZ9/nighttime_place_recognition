import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

list_features = np.zeros((5, 24), dtype=int)
pics_num = np.zeros((1,24), dtype=int)

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]#点的横坐标
k1 = np.random.rand(24)
k2 = np.random.rand(24)
k3 = np.random.rand(24)
k4 = np.random.rand(24)
k5 = np.random.rand(24)

print(k1)
plt.plot(x,k1,color = '#00ae9d',label="Harris Corner")
plt.plot(x,k2,color = '#f7acbc',label="Shi-Tomasi")
plt.plot(x,k3,color = '#7f7522',label="SIFT")
plt.plot(x,k4,color = '#f58220',label="SURF")
plt.plot(x,k5,color = '#6950a1',label="ORB")

plt.title('The Number of Feature Points Detected at Different Time')
plt.xticks(x)
plt.xlabel("Hour")
plt.ylabel("Number of Feature Points")
plt.legend(loc = "best")
plt.show()
