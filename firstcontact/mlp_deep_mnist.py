import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_input = 784
n_hidden1 = 512
n_hidden2 = 512
n_hidden3 = 256
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")

stddev = 0.05
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=stddev)),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=stddev)),
    'h3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden3, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'b3': tf.Variable(tf.random_normal([n_hidden3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    layer1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3']))
    droplayer = tf.nn.dropout(layer3, _keep_prob)
    return (tf.matmul(droplayer, _weights['out'])+ _biases['out'])

pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)
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
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 0.6}
        sess.run(optm, feed_dict=feeds)
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 1.0}
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch


    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, dropout_keep_prob: 1.0}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Train accuracy: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels, dropout_keep_prob: 1.0}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Test accuracy: %.3f" % (test_acc))

