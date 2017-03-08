import numpy as np
from numpy import *
import matplotlib.pyplot as plt

dataSet = [[-0.017612, 14.053062], [-1.395634, 4.662541], [-0.752157, 6.538620], [0.423363, 11.054677],[0.406704, 7.067335], [0.667394, 12.741452], [-2.460150, 6.866805], [0.569411, 9.548755], [-0.026632, 10.427743], [0.850433, 6.920334], [1.347183, 13.175500], [1.176813, 3.167020], [-1.781871, 9.097953]]

dataMat = mat(dataSet).T
plt.scatter(dataMat[0], dataMat[1], c='red', marker='o')

X = np.linspace(-2,2,100)
Y = 2.8*X +9
plt.plot(X,Y)
plt.show()
print(linalg.matrix_rank(mat(dataSet)))

