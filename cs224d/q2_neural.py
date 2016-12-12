import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    ofs = 0
    Dx, H, Dy = (dimensions[0],dimensions[1],dimensions[2])
    W1 = np.reshape(params[ofs:ofs + Dx * H],(Dx,H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H],(1,H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H*Dy],(H,Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy],(1,Dy))

    #forward propagation
    hidden = sigmoid(data.dot(W1) + b1)
    prediction = softmax(hidden.dot(W2)+b2)
    cost = -np.sum(np.log(prediction)*labels)

    # backward propagation
    delta = prediction - labels
    gradW2 = hidden.T.dot(delta)
    gradb2 = np.sum(delta,axis = 0)
    delta = delta.dot(W2.T)*sigmoid_grad(hidden)
    gradW1 = data.T.dot(delta)
    gradb1 = np.sum(delta, axis=0)

    #stack gradients
    grad = np.concatenate((gradW1.flatten(),gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    return cost, grad

def sanity_check():
    print("Running sanity check...")

    N = 20
    dimensions = [10,5,10]
    data = np.random.randn(N,dimensions[0])
    labels = np.zeros((N,dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0]+1)*dimensions[1]+(dimensions[1]+1)*dimensions[2],)

    gradcheck_naive(lambda params: forward_backward_prop(data,labels,params,dimensions),params)
    print("sanity check passed")

if __name__ == "__main__":
    sanity_check()
