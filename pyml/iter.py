# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
#from common_libs import *
import matplotlib.pyplot as plt

A = mat([[8,-3,2],[4,11,-1],[6,3,12]])
b = mat([20,33,36])
result = linalg.solve(A,b.T)
print(result)

error = 1.0e-6
steps = 100
xk = zeros((3,1))
errorlist = []
B0= (diag(A)*eye(shape(A)[0])-A)/diag(A)
f=b/diag(A)
for k in range(steps):
    xk_1 = xk
    xk = B0*xk+f 
    errorlist.append(linalg.norm(xk-xk_1))
    if errorlist[-1]<error:
        print(k+1)
        break
print(xk)
    
