import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

stddev = 0.1
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
    'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=stddev)),
    'wd2': tf.Variable(tf.random_normal([1024, n_classes], stddev=stddev)),
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
    'bd2': tf.Variable(tf.random_normal([n_classes], stddev=stddev)),
}

def conv_basic(_X, _weights, _biases, _keep_prob):
    _input_image = tf.reshape(_X, shape=[-1, 28, 28, 1])
    _conv1 = tf.nn.conv2d(_input_image, _weights['wc1'], strides=[1,1,1,1], padding="SAME")
    _mean, _var = tf.nn.moments(_conv1, [0,1,2])
    _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _biases['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keep_prob)

    _conv2 = tf.nn.conv2d(_pool_dr1, _weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv2, [0,1,2])
    _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _biases['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keep_prob)

    _dense1 = tf.reshape(_pool_dr2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _weights['wd1']), _biases['bd1']))
    _fc1_dr1 = tf.nn.dropout(_fc1, _keep_prob)
    _out = tf.add(tf.matmul(_fc1_dr1, _weights['wd2']), _biases['bd2'])
    out = {
        'input_r': _input_image, 'conv1': _conv1, 'pool1': _pool1, 'pool_dr1': _pool_dr1,
        'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
        'fc1': _fc1, 'fc_dr1': _fc1_dr1, 'out': _out
    }
    return out

pred = conv_basic(x, weights, biases, keep_prob)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
training_epochs = 15
batch_size = 100
display_step = 1
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys, keep_prob: 0.7}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, keep_prob: 1.0}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Train accuracy: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Test accuracy: %.3f" % (test_acc))
