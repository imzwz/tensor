import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from input_data import input_data

mnist = input_data.read_data_sets('MNIST', one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_output = 784

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output])),
}

def dae(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out'])+_biases['out'])

recon = dae(x, weights, biases, dropout_keep_prob)

cost = tf.reduce_mean(tf.pow(recon-y, 2))
optm = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

savedir = "nets/"
saver = tf.train.Saver(max_to_keep=1)

TRAIN_FLAG = 1
epochs = 50
batch_size = 100
disp_step = 10

sess = tf.Session()
sess.run(init)

if TRAIN_FLAG:
    for epoch in range(epochs):
        num_batch = int(mnist.train.num_examples/batch_size)
        total_cost = 0
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + 0.3*np.random.randn(batch_size, 784)
            feeds = { x: batch_xs_noisy, y: batch_ys, dropout_keep_prob: 1.0}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)

        if epoch % disp_step == 0:
            print("display")

            

