import numpy as np
import tensorflow as tf
def softmax(x):
    maxes = tf.expand_dims(tf.reduce_max(x,reduction_indices=[1]),1)
    x_red = x - maxes
    x_exp = tf.exp(x_red)
    sums = tf.expand_dims(tf.reduce_sum(x_exp,reduction_indices=[1]),1)
    out = x_exp /sums
    return out

def cross_entropy_loss(y,yhat):
    y = tf.to_float(y)
    out = -tf.reduce_sum(y*tf.log(yhat))
    return out

def test_softmax_basic():
    print("Running basic tests...")
    test1 = softmax(tf.convert_to_tensor(np.array([[1001,1002],[3,4]]),dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(tf.convert_to_tensor(np.array([[-1001,-1002]]),dtype=tf.float32))
    with tf.Session():
        test2 = test2.eval()
    assert np.amax(np.fabs(test2 - np.array([0.73105858,0.26894142]))) <= 1e-6
    
    print("Basic sotfmax tests passed")

def test_cross_entropy_loss_basic():
    y = np.array([[0,1],[1,0],[1,0]])
    yhat = np.array([[.5,.5],[.5,.5],[.5,.5]])
    test1 = cross_entropy_loss(tf.convert_to_tensor(y,dtype=tf.int32), tf.convert_to_tensor(yhat,dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    result = -3*np.log(.5)    
    assert np.amax(np.fabs(test1 - result)) <= 1e-6
    print("Basic cross-entropy tests pass")

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()


