import sklearn.preprocessing as preprocessing
import tensorflow as tf
import numpy as np

singleValue = tf.Variable(initial_value=2.1)
oneDimValue = tf.Variable(initial_value=np.array([1, 6, 7]))
twoDimValue = tf.Variable(initial_value=np.array([[3, 6, 1], [3, 3, 2], [6, 9, 8]], np.float32))
twoDimValueList = [[3, 6, 1], [3, 3, 2], [6, 9, 8]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(twoDimValue))

    print("===================== L1 ======================")
    normalized_l1 = preprocessing.normalize(twoDimValueList, norm='l1')
    print(normalized_l1)
    print("===================== L2 ======================")
    normalized_l2 = preprocessing.normalize(twoDimValueList, norm='l2')
    print(normalized_l2)
    print("===================== MAX ======================")
    normalized_max = preprocessing.normalize(twoDimValueList, norm='max')
    print(normalized_max)




