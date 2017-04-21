import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_hidden1 = 256
n_hidden2 = 128
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

stddev = 0.1
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(_X, _weights, _biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2']))
    return (tf.matmul(layer2, _weights['out'])+ _biases['out'])

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
training_epochs = 20
batch_size = 100
display_step = 5
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Train accuracy: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Test accuracy: %.3f" % (test_acc))
