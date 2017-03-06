import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import LanguageModel
from utils import calculate_perplexity

class Config(object):
    batch_size = 64
    embed_size = 50
    hidden_size = 100
    num_steps = 10
    max_epochs = 16
    early_stopping = 2
    dropout = 0.9
    lr = 0.001

class RNNLM_Model(LanguageModel):
    def load_data(self, debug=False):
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)


