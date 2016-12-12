import numpy as np
import random

def softmax(x):
    if len(x.shape) > 1:
        tmp = np.max(x, axis = 1)
        x -= tmp.reshape((x.shape[0],1))
        x = np.exp(x)
        tmp = np.sum(x,axis = 1)
        x /= tmp.reshape((x.shape[0],1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

def test_softmax_basic():
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1-np.array([0.26894142,0.73105858]))) <= 1e-6
    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test1-np.array([[0.26894142,0.73105858],[0.26894142,0.73105858]]))) <= 1e-6
    print("verify passed")

if __name__== "__main__":
    #__name__ 是当前模块名，当模块被直接运行时模块名为 __main__ 。这句话的意思就是，当模块被直接运行时，代码将被运行，当模块是被导入时，代码不被运行。
    test_softmax_basic()
    
