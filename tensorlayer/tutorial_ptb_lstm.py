import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

flags = tf.flags
flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.model == "small":
        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000
    elif FLAGS.model == "medium":
        init_scale = 0.05
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 35
        hidden_size = 650
        max_epoch = 6 
        max_max_epoch = 39 
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
        vocab_size = 10000
    elif FLAGS.model == "large":
        init_scale = 0.04
        learning_rate = 1.0
        max_grad_norm = 10
        num_layers = 2
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14 
        max_max_epoch = 55 
        keep_prob = 0.35
        lr_decay = 1/1.15
        batch_size = 20
        vocab_size = 10000
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
    print('len(train_data) {}'.format(len(train_data)))
    print('len(valid_data) {}'.format(len(valid_data)))
    print('len(test_data) {}'.format(len(test_data)))
    print('vocab_size {}'.format(vocab_size))

    sess = tf.InteractiveSession()

    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    input_data_test = tf.placeholder(tf.int32, [1,1])
    targets_test = tf.placeholder(tf.int32, [1,1])

    def inference(x, is_training, num_steps, reuse=None):
        print("\nnum_steps : %d, is_training : %s, reuse : %s" % (num_steps, is_training, reuse))

        initializer = tf.random_uniform_initializer(init_scale, init_scale)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.EmbeddingInputlayer(inputs = x, vocabulary_size = vocab_size, embedding_size = hidden_size, E_init = tf.random_uniform_initializer(-init_scale, init_scale), name='embedding_layer')

