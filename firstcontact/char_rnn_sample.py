import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle

load_dir = "data/linux_kernel"
with open(os.path.join(load_dir, 'chars_vocab.pkl'), 'rb') as f:
    chars, vocab = cPickle.load(f)
vocab_size = len(vocab)

rnn_size = 128
num_layers = 2
batch_size = 1
seq_length = 1

unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell = tf.nn.rnn_cell.MultiRNNCell([unitcell]* num_layers)
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
istate = cell.zero_state(batch_size, tf.float32)

with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
    inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
    inputs = [tf.squeeze(_input, [1]) for _input in inputs]

def loop(prev, _):
    prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)

outputs, final_state = seq2seq.rnn_decoder(inputs, istate, cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs = tf.nn.softmax(logits)