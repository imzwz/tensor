import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.002
training_epochs = 50
batch_size = 100
display_step = 1

n_input = 784
n_hidden1 = 256
n_hidden2 = 256
n_hidden3 = 256
n_hidden4 = 256
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder("float")

stddev = 0.1
weights = {
    'h1': tf.get_variable("h1", shape=[n_input, n_hidden1], initializer=xavier_init(n_input, n_hidden1)),
    'h2': tf.get_variable("h2", shape=[n_hidden1, n_hidden2], initializer=xavier_init(n_hidden1, n_hidden2)),
    'h3': tf.get_variable("h3", shape=[n_hidden2, n_hidden3], initializer=xavier_init(n_hidden2, n_hidden3)),
    'h4': tf.get_variable("h4", shape=[n_hidden3, n_hidden4], initializer=xavier_init(n_hidden3, n_hidden4)),
    'out': tf.get_variable("out", shape=[n_hidden4, n_classes], initializer=xavier_init(n_hidden4, n_classes)),

}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'b3': tf.Variable(tf.random_normal([n_hidden3])),
    'b4': tf.Variable(tf.random_normal([n_hidden4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(_X, _weights, _biases, _keep_prob):
    layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), _keep_prob)
    layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), _keep_prob)
    layer3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), _keep_prob)
    layer4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer3, _weights['h4']), _biases['b4'])), _keep_prob)
    return (tf.matmul(layer4, _weights['out'])+ _biases['out'])
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

