import tensorflow as tf
import numpy as np

weight = tf.Variable(initial_value=[[3., 9., 1.], [9., 9., 8.], [1., 5., 7.]], name='var_01')
value = tf.Variable(initial_value=[[2., 2., 5.], [9., 4., 1.], [6., 6., 0.]], name='value')

np_weight = tf.Variable(initial_value=np.array([[3., 9., 1.], [9., 9., 8.], [1., 5., 7.]]), dtype=np.float32)
np_value = tf.Variable(initial_value=np.array([[2., 2., 5.], [9., 4., 1.], [6., 6., 0.]]), dtype=tf.float32)

eyeMat = tf.Variable(tf.eye(3, 3), 'eye')
y = tf.matmul(np_weight, np_value)
y_transposed = tf.transpose(y)
y_transposed = tf.transpose(y,
                            perm=[1, 0],
                            name='TransposeOperation_Fully_Declared')

y_divided = tf.div(y, np.array([[1., 5., 10.], [10., 3., 3.], [2., 5., 7.]], dtype=np.float32), name='Mat_division_operation')
z = tf.sigmoid(tf.add(y, weight)) * 9 + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(y))
    print("Transposed")
    print(sess.run(y_transposed))
    print("y_divided")
    print(sess.run(y_divided))