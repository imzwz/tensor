import time
import math
import numpy as np
import tensorflow as tf
from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils import data_iterator

class Config(object):
    batch_size = 64
    n_samples = 1024
    n_features = 100
    n_classes = 5
    max_epochs = 50
    lr = 1e-4

class SoftmaxModel(Model):
    def load_data(self):
        np.random.seed(1234)
        self.input_data = np.random.rand(self.config.n_samples,self.config.n_features)
        self.input_labels = np.ones((self.config.n_samples,),dtype=np.int32)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size,self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.n_classes))

    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch, 
        }
        return feed_dict

    def add_training_op(self,loss):
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0, name = 'global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step = global_step)
        return train_op

    def add_model(self, input_data):
        n_features, n_classes = self.config.n_features, self.config.n_classes
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(tf.zeros([n_features,n_classes]), name = 'weights')
            biases = tf.Variable(tf.zeros([n_classes]), name='biases')
            logits = tf.matmul(input_data, weights) + biases
            out = softmax(logits)
        return out

    def add_loss_op(self, pred):
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        return loss

    def run_epoch(self, sess, input_data, input_labels):
        average_loss = 0
        for step, (input_batch, label_batch) in enumerate(data_iterator(input_data,input_labels,batch_size = self.config.batch_size, label_size=self.config.n_classes)):
            feed_dict = self.create_feed_dict(input_batch, label_batch)
            _, loss_value = sess.run([self.train_op, self.loss],feed_dict = feed_dict)
            average_loss += loss_value

        average_loss = average_loss / step
        return average_loss

    def fit(self, sess, input_data, input_labels):
        losses = []
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, input_data, input_labels)
            duration = time.time() - start_time
            print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, average_loss, duration))
            losses.append(average_loss)
        return losses

    def __init__(self,config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def test_SoftmaxModel():
        config = Config()
        with tf.Graph().as_default():
            model = SoftmaxModel(config)
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            losses = model.fit(sess, model.input_data,model.input_labels)

        assert losses[-1] < .5
        print("Basic classifier tests pass")

if __name__ == '__main__':
    tset_SoftmaxModel()




