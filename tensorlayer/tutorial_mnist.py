import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time
#prepare data
sess = tf.InteractiveSession()
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))
#for convolutional neural network, the data can be load by:
#X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,28,38,1))
#define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

#define the network
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')
y = network.outputs

#define cost function and metric
cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y,1),y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y),1)

#define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

#initialize all variables
sess.run(tf.initialize_all_variables())
#print network information
network.print_params()
network.print_layers()

#train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_, acc = acc, batch_size=500, n_epoch=500, print_freq=5, X_val=X_val, y_val=y_val, eval_train=False)
#evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)
tl.files.save_npz(network.all_params, name='model.npz')
sess.close()






