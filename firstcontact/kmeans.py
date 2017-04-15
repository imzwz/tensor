import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import tensorflow as tf

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])
#conjunto_puntos = [[x0,y0],[x1,y1],...,[x1999,y1999]]

#df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos], "y": [v[1] for v in conjunto_puntos]})
#sns.lmplot("x", "y", data= df, fit_reg = False, size=6)
#plt.show()

vectors = tf.constant(conjunto_puntos)
#print(vectors.get_shape()) -> [2000,2]
k = 6 
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
#print(centroides.get_shape()) -> [4,2]
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
#print(expanded_vectors.get_shape()) -> [1, 2000, 2]
#print(expanded_centroides.get_shape()) -> [4, 1, 2]

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), reduction_indices = [1]) for c in range(k)])
update_centroides = tf.assign(centroides, means)
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
    data["x"].append(conjunto_puntos[i][0])
    data["y"].append(conjunto_puntos[i][1])
    data["cluster"].append(assignment_values[i])
df = pd.DataFrame(data)
sns.lmplot("x","y", data =df , fit_reg = False, size=6, hue="cluster", legend=False)
plt.show()



