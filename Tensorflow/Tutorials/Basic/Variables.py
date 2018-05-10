import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])

x_data = [[1, 6, 7], [6, 4, 2]]

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

expression = tf.matmul(X, W) + b

session = tf.Session()

session.run(tf.global_variables_initializer())

print(X)
print(W)
print(b)

print(session.run(expression, feed_dict={X : x_data}))