import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves cPickle

data_dir = "data/linux_kernel"
save_dir = "data/linux_kernel"
input_file = os.path.join(data_dir, "input.txt")
with open(input_file, "r") as f:
    data = f.read()

counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
print( "Type of 'counter.items()' is %s and length is %d" % (type(counter.items()), len(counter.items())))

for i in range(5):
    print("[%d/%d]" % (i, 3)),
    print(list(counter.items())[i])

chars, counts = zip(*count_pairs)
vocab = dict(zip(chars, range(len(chars))))
print( "Type of 'chars' is %s and length is %d" % (type(chars), len(chars))))
#print( "Type of '' is %s and length is %d" % (type(), len())))
for i in range(5):
    print("[%d/%d]" % (i, 3)),
    print("chars[%d] is %s" % (i, chars[i]))

print( "Type of 'vocab' is %s and length is %d" % (type(vocab), len(vocab))))
for i in range(5):
    print("[%d/%d]" % (i, 3)),
    print("vocab[%d] is %s" % (chars[i], vocab[chars[i]]))

with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'wb') as f:
    cPickle.dump((chars, vocab), f)

corpus = np.array(list(map(vocab.get, data)))
print( "Type of 'corpus' is %s, shape is %s and length is %d" % (type(corpus), corpus.shape, len(corpus))))
ckeck_len = 10
for i in range(ckeck_len):
    _wordidx = corpus[i]
    #print( "Type of '' is %s and length is %d" % (type(), len())))

batch_size = 50
seq_length = 200
num_batchs = int(corpus.size / (batch_size * seq_length))
corpus_reduced = corpus[:(num_batchs*batch_size*seq_length)]
xdata = corpus_reduced
ydata = np.copy(xdata)
ydata[:-1] = xdata[1:]
ydata[-1] = xdata[0]

xbatches = np.split(xdata.reshape(batch_size, -1), num_batchs, 1)
ybatches = np.split(ydata.reshape(batch_size, -1), num_batchs, 1)
nbatch = 5
temp = xbatches[0:nbatch]

for i in range(nbatch):
    temp2 = temp[i]

vocab_size = len(vocab)
rnn_size = 128
num_layers = 2
grad_clip = 5.

unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell = tf.nn.rnn_cell.MultiRNNCell([unitcell]* num_layers)
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets = tf.placeholder(tf.int32, [batch_size, seq_length])
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

outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, istate, cell, loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs = tf.nn.softmax(logits)

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1],
                                        [tf.ones([batch_size*seq_length])], vocab_size)])

cost = tf.reduce_sum(loss) / batch_size / seq_length
final_state = last_state
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
_optm = tf.train.AdamOptimizer(lr)
optm = _optm.apply_gradients(zip(grads, tvars))

num_epochs = 50
save_every = 500
learning_rate = 0.002
decay_rate = 0.97

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.train.SummaryWriter(save_dir, graph=sess.graph)
saver = tf.train.Saver(tf.all_variables)
init_time = time.time()
for epoch in range(num_epochs):
    sess.run(tf.assign(lr, learning_rate*(decay_rate**epoch)))
    state = sess.run(istate)
    batch_idx = 0
    for iteration in range(num_batchs):
        start_time = time.time()
        randbatchidx = np.random.randint(num_batchs)
        xbatch = xbatches[batch_idx]
        ybatch = ybatches[batch_idx]
        batch_idx = batch_idx+1

        train_loss, state, _ = sess.run([cost, final_state, optm],
                                feed_dict={input_data: xbatch, targets: ybatch, istate: state})
        total_iter = epoch*num_batchs + iteration
        end_time = time.time()
        duration = end_time - start_time

        if total_iter % 100 == 0:
            print()
        if total_iter % save_every == 0:
            ckpt_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=total_iter)

