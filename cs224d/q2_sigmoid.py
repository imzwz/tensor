import numpy as np

def sigmoid(x):
    x = 1/(1+np.exp(-x))
    return x

def sigmoid_grad(f):
    f = f*(1-f)
    return f

def test_sigmoid_basic():
    print("Running basic test...")
    x = np.array([[1,2],[-1,-2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    assert np.amax(f-np.array([[0.73105858,0.88079708],[0.26894142,0.11920292]])) <= 1e-6
    print(g)
    print("verify test passed")

if __name__ == "__main__":
    test_sigmoid_basic()
