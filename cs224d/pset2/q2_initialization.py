import numpy as np
import tensorflow as tf

def xavier_weight_init():
    def _xavier_initializer(shape, **kwargs):
        m = shape[0]
        n = shape[1] if len(shape) > 1 else shape[0]
        bound = np.sqrt(6) / np.sqrt(m+n)
        out = tf.random_uniform(shape, minval=-bound,maxval = bound)
        return out

    return _xavier_initializer

def test_initialization_basic():
    print("Running basic tests")
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape

    print("Basic Xavier initialization tests pass")

def test_initialization():
    print("Running your tests...")
    raise NotImplementdError

if __name__ == "__main__":
    test_initialization_basic()

