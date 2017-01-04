import collections
import math
import os
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import tensorlayer as tl
import time

flags = tf.flags
flags.DEFINE_string("model", "one", "A type of model.")
FLAGS = flags.FLAGS

def main_word2vec_basic():
    ##step 1
    words = tl.files.load_matt_mahoney_text8_dataset()
    data_size = len(words)
    print('Data size', data_size)
    resume = False
    _UNK = "_UNK"

    if FLAGS.model == "one":
        vocabulary_size = 50000
        batch_size = 128
        embedding_size = 128
        skip_window = 1
        num_skips = 2

        num_sampled = 64
        learning_rate = 1.0
        n_epoch = 20
        model_file_name = "model_word2vec_50k_128"
    
    if FLAGS.model == "two":
        vocabulary_size = 80000
        batch_size = 20
        embedding_size = 200
        skip_window = 5
        num_skips = 10
        num_sampled = 100
        learning_rate = 0.2
        n_epoch = 15
        model_file_name = "model_word2vec_80k_200"

    if FLAGS.model == "three":
        vocabulary_size = 80000
        batch_size = 20
        embedding_size = 200
        skip_window = 5
        num_skips = 10
        num_sampled = 25
        learning_rate = 0.025
        n_epoch = 20
        model_file_name = "model_word2vec_80k_200_opt"

    if FLAGS.model == "four":
        vocabulary_size = 80000
        batch_size = 100
        embedding_size = 600
        skip_window = 5
        num_skips = 10
        num_sampled = 25
        learning_rate = 0.03
        n_epoch = 200*10
        model_file_name = "model_word2vec_80k_600"

    num_steps = int((data_size/batch_size) * n_epoch)
    print('%d Steps a Epoch, total Epochs %d' % (int(data_size/batch_size), n_epoch))
    print('  learning_rate: %f' % learning_rate)
    print('  batch_size: %d' % batch_size)

    ##step 2
    if resume:
        print("load existing data and dictionaries" + "!"*10)
        all_var  = tl.files.load_npy_to_any(name=model_file_name+'.npy')
        data = all_var['data']; count = all_var['count']
        dictionary = all_var['dictionary']
        reverse_dictionary = all_var['reverse_dictionary']
    else:
        data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size True, _UNK)
    print('Most 5 common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    del words

    ##step3
    print()
    data_index = 0
    batch, labels, data_index = tl.nlp.generate_skip_gram_batch(data=data, batch_size=20, num_skips=4, skip_window=2, data_index=0)
    for i in range(20):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i,0]])
    
    ##step4
    print()
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace = False)
    print_freq = 2000

    ##step5
    print()
    sess.run(tf.initialize_all_variables())
    if resume:
        print("Load existing model" + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name+ '.ckpt')

