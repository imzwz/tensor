import sys
import time
import numpy as np
from copy import deepcopy
from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_terator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

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
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array([self.vocab.encode(word) for word in get_ptb_dataset('train')],dtype=np.int32)
        self.encoded_test = np.array([self.vocab.encode(word) for word in get_ptb_dataset('test')],dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
            imputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]
            return inputs
    
    def add_prediction(self, rnn_outputs):
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix',[self.config.hidden_size, len(self.vocab)])
            proj_b = tf.get_variable('Bias', [len(self.vocab)])
            outputs = [tf.matmul(o,U) + proj_b for o in rnn_outputs]
        return outputs

    def add_loss_op(self, output):
        add_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        cross_entropy = sequence_loss([output], [tf.reshape(self.labels_placeholder, [-1])], add_ones, len(self.vocab))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.calculate_loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        output = tf.reshape(tf.concat(1,self.outputs), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_model(self, inputs):
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]
        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('HMatrix',[self.config.hidden_size, self.config.hidden_size])
                RNN_I = tf.get_variable('IMatrix',[self.config.embed_size, self.config.hidden_size])
                RNN_b = tf.get_variable('B',[self.config.hidden_size])
                state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I)+RNN_b)
                rnn_outputs.append(state)
            self.final_state = rnn_outputs[-1]
        
        with tf.variable_scope('RNNDropout'):
            rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]
        return rnn_outputs

    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x,y) in enumerate(ptb_iterator(data, config.batch_size, config.num_steps)):
            feed = {self.input_placeholder: x, self.labels_placeholder: y, self.initial_state: state, self.dropout_placeholder: dp}
            loss, state, _ = session.run([self.calculate_loss, self.final_state, train_op], feed_dict= feed)
            total_loss.append(loss)
            if verbose and step $ verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')

        return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    state = model.initial_state.eval()
    tokens = [model.vocab.endoce(word) for word in starting_text.split()]
    for i in range(stop_length):
        feed = {model.input_placeholder: [tokens[-1:]], model.initial_state: state, model.dropout_placeholder: 1}
        state, y_pred = session.run( [model.final_state, model.predictions[-1]], feed_dict= feed)
        next_word_idx = sample(y_pred[0], temperature=temp)
        tokens.append(next_word_idx)
        if stop_tokens and modle.vocab.decode(tokens[-1]) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]

    return output

def generate_sentence(session, model, config, *args, **kwargs):
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
    config = Config()
    gen_config = deepcope(config)
    gen_config.batch_size = gen_config.num_steps = 1

if __name__ == "__main__":
    test_RNNLM()
    
                
        
        




