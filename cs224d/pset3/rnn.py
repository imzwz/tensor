import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow
import tree as tr
from utils import Vocab

RESET_AFTER = 50
class Config(object):
    embed_size = 35
    label_size = 2
    early_stopping = 2 
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 30
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)

class RNN_Model():
    def load_data(self):
        self.train_data, self.dev_data, self.test_data = tr.simplied_data(700, 100, 200)
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))
    
    def inference(self, tree, predict_only_root=False):
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.iteritems() if node.label!=2]
            node_tensors = tf.concat(0, node_tensors)
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        with tf.variable_scope('Composition'):
            tf.get_variable('embedding', [self.vocab.total_words, self.config.embed_size])
            tf.get_variable('W1', [2*self.config.embed_size, self.config.embed_size])
            tf.get_variable('b1', [1, self.config.embed_size])
        with tf.variable_scope('Projection'):
            tf.get_variable('U', [self.config.embed_size, self.config.label_size])
            tf.get_variable('bs',[1,self.config.label_size])
    
    def add_model(self, node):
        with tf.variable_scope('Composition', reuse=True):
            embedding = tf.get_variable('embedding')
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')
        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf:
            word_id = self.vocab.encode(node.word)
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, word_id), 0)
        else:
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))
            child_tensor = tf.concat(1, [node_tensors[node.left], node_tensors[node.right]])
            curr_node_tensor = tf.nn.relu(tf.matmul(child_tensor, W1)+b1)
        node_tensors[node] = curr_node_tensor
        return node_tensors
    
    def add_projections(self, node_tensors):
        logits = None
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
            bs = tf.get_variable('bs')
            logits = tf.matmul(node_tensors, U)
            logits += bs
        return logits

    def loss(self, logits, labels):
        loss = None
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
        l2loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)
        cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        loss = cross_entropy + self.config.l2* l2loss
        return loss
    def training(self, loss):
        train_op = None
        optimizer = tf.train.GraidientDescentOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op
    
    def predictions(self, y):
        predictions = None
        predictions = tf.argmax(y, 1)
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss=False):
        results = []
        losses = []
        for i in range(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+!)*RESET_AFTER]:
                    logits = self.inference(tree, True)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if get_loss:
                        roto_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label]))
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model=False, verbose=True):
        step = 0
        loss_history = []
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.initialize_all_variables()
                    sess.run(init)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                for _ in range(RESET_AFTER):
                    if step>= len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree)
                    labels = [l for l in tree.labels if l!=2]
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} : loss = {}'.format(step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists('./weights'):
                    os.makedirs('./weights')
                saver.save(sess, './weights/%s.temp'%self.config.model_name)
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name)
        ##
        
    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):

            print('epoch %d'%epoch)

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2,2])
        for l,p in itertools.izip(labels, predictions):
            confmat[l,p] += 1
        return confmat    

def test_RNN():
    config = Config()

if __name__ == "__main__":
    test_RNN()





