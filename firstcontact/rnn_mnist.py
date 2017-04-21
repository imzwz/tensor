import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

trainimgs = mnist.train.images
testimgs = mnist.test.images
trainlabels = mnist.train.labels
testlabels = mnist.test.labels

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
n_classes = trainlabels.shape[1]

diminput = 28
dimhidden = 128
dimoutput = n_classes
nsteps = 28
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))

}
x = tf.placeholder("float", [None, nsteps, diminput])
istate = tf.placeholder("float", [None, 2*dimhidden])
y = tf.placeholder("float", [None, dimoutput])

stddev = 0.1
'''
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
'''
def RNN_network(_X, _istate, _W, _b, _nsteps, _name):
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, diminput])
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    _Hsplit = tf.split(0, _nsteps, _H)
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, initial_state=_istate)
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit, 'LSTM_O': _LSTM_O,
        'LSTM_S': _LSTM_S, 'O': _O
    }

myrnn = RNN_network(x, istate, weights, biases, nsteps, 'basic')
pred = myrnn['O']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

init = tf.global_variables_initializer()
training_epochs = 5
batch_size = 120
display_step = 1
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph = sess.graph)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)/total_batch

    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*dimhidden))}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Train accuracy: %.3f" % (train_acc))
        testimgs = testimgs.reshape((ntest, nsteps, diminput))
        feeds = {x: testimgs, y: mnist.test.labels, istate: np.zeros((ntest, 2*dimhidden))}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Test accuracy: %.3f" % (test_acc))
