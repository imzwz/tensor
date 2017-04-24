import collections
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

batch_size = 20
embedding_size = 2
num_sampled = 15

sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

words = ' '.join(sentences).split()

count = collections.Counter(words).most_common()

rdic = [i[0] for i in count]
dic = {w: i for i, w in enumerate(rdic)}
voc_size = len(dic)

data = [dic[word] for word in words]

cbow_pairs = [];
for i in range(1, len(data)-1):
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]])
skip_gram_pairs = [];
for c in cbow_pairs:
    skip_gram_pairs.append([c[1], c[0][0]])
    skip_gram_pairs.append([c[1], c[0][1]])

def generate_batch(size):
    assert size < len(skip_gram_pairs)
    x_data = []
    y_data = []
    r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pairs[i][0])
        y_data.append(skip_gram_pairs[i][1])
    return x_data, y_data

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.device('/cpu:0'):
    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, voc_size))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(3000):
        batch_inputs, batch_labels = generate_batch(batch_size)
        _, loss_val = sess.run([train_op, loss], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
        if step % 500 == 0:
            print("loss at %d: %.5f" % (step, loss_val))
        trained_embeddings = embeddings.eval()

if trained_embeddings.shape[1] == 2:
    labels = rdic[:20]
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i, :]
        plt.scatter(x,y)
        plt.annotate(label, xy=(x,y), xytest=(5,2), textcoords='offset points', ha='right', va='bottom')
    plt.show()