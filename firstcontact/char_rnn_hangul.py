import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle
from TextLoader import *
from Hangulpy import *

data_dir = "data/nine_dreams"
batch_size = 50
seq_length = 50
data_loader = TextLoader(data_dir, batch_size, seq_length)

vocab_size = data_loader.vocab_size
vocab = data_loader.vocab
chars = data_loader.chars

x, y = data_loader.next_batch()

rnn_size = 512
num_layers = 3
grad_clip = 5.
vocab_size = data_loader.vocab_size

unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)

input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets = tf.placeholder(tf.int32, [batch_size, seq_length])
initial_state = cell.zero_state(batch_size, tf.float32)

with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
        inputs = [tf.squeeze(input_ [1]) for input_ in inputs]

outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs = tf.nn.softmax(logits)

loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                    [tf.reshape(targets, [-1])],
                                    [tf.ones([batch_size * seq_length])],
                                    vocab_size)

cost = tf.reduce_sum(loss) / batch_size / seq_length

lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm = tf.train.AdamOptimizer(lr)
optm = _optm.apply_gradients(zip(grads, tvars))
final_state = last_state

num_epochs = 500
save_every = 1000
learning_rate = 0.0002
decay_rate = 0.97

save_dir = 'data/nine_dreams'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.train.SummaryWriter(save_dir, graph=sess.graph)
saver = tf.train.Saver(tf.all_variables)

for epoch in range(num_epochs):
    sess.run(tf.assign(lr, learning_rate * (decay_rate ** epoch)))
    data_loader.reset_batch_pointer()
    state = sess.run(initial_state)
    for b in range(data_loader.num_batches):
        start = time.time()
        x, y = data_loader.num_batches()
        feed = {input_data: x, targets: y, initial_state: state}
        train_loss, state, _ = sess.run([cost, final_state, optm], feed)
        end = time.time()

        if b % 100 == 0:
            print()
        if (epoch * data_loader.num_batches + b) % save_every == 0:
            chechpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, chechpoint_path, global_step= epoch* data_loader.num_batches + b)
            print()