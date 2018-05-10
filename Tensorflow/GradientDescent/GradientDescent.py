import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = [[1., 2., 3.], [5., 6., 7.]]
Y = [[1., 2., 3.]]
X = np.asmatrix(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

print(X)
print(Y)

m = n_samples = len(X)

W = tf.placeholder(np.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m

W_val = []
cost_val = []

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(-300, 500):
    print (i * 0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()