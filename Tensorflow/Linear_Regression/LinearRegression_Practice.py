import tensorflow as tf
from tensorflow.contrib.data import Dataset
import numpy as np
import pandas as pd
import csv

# Country - Area data in a form separated by \t

TRAIN_DATA = [] # 180
TRAIN_LABEL = []
TEST_DATA = []  # 21
TEST_LABEL = []
COMPLETE_DATA = []

# Prepare our data to train & test
def data_preparation():
    #One hot
    train_data = np.array(np.zeros(1))
    train_label = np.array(np.zeros(1))
    test_data = np.array(np.zeros(1))
    test_label = np.array(np.zeros(1))
    complete_data = []

    data_size = 0
    stripped = []

    train_data: pd.DataFrame = pd.read_csv(filepath_or_buffer='F:\\2018_Spring\Programming\Python\Tensorflow\Data\country-pop.csv',
                             delimiter='\t', usecols=[6, 10], header=None, names=['VAR1', 'VAR2'], na_values=("N.A.", np.nan), thousands=',')

    # train_data['VAR2'] = ['' if '%' in x else x for x in train_data['VAR2']]
    # for idx in range(train_data.size):
        # print(train_data['VAR2'].iterrows())
        # train_data['VAR2'][idx] = train_data['VAR2'][idx].replace("%", "")
    # series: pd.Series = train_data['VAR2']

    fill_values = {'VAR2': 50}
    train_data.fillna(value=fill_values, inplace=True)
    idx = 0
    for x in train_data.VAR2:
        x = str(x).replace(" %", "")
        train_data.VAR2[idx] = x
        idx += 1

    # train_data.replace(to_replace={'VAR2':{'%': ''}}, inplace=True)
    # train_data['VAR2'].replace('[\%)]', '', inplace=True)

    """
    for idx in range(series.size):
        if pd.isna(series[idx]):
            series[idx] = 50
        else:
            series[idx] = series[idx].replace("%", "")
    """

    train_label = pd.read_csv(filepath_or_buffer='F:\\2018_Spring\Programming\Python\Tensorflow\Data\country-pop.csv',
                              delimiter='\t', usecols=[2], header=None, names=['POPULATION'], thousands=',')
    # print(train_data)


    return train_data, train_label

# Data
# train_samplesize, test_samplesize = 0
TRAIN_DATA, TRAIN_LABEL = data_preparation()

TRAIN_DATA = np.array(TRAIN_DATA)
TRAIN_LABEL = np.asarray(TRAIN_LABEL)
print(TRAIN_DATA)
# TRAIN_DATA = [[3., 4.], [5., 1.], [9., 4.]]
# TRAIN_LABEL = [[4.], [1.], [9.]]

# Parameters
learning_rate = 0.001
epoch = 1000
display_step = 50

# tf Graph Input
X = tf.placeholder(tf.float32, [2, 1])
Y = tf.placeholder(tf.float32, [1, 1])

# Train data
train_x = np.asarray(TRAIN_DATA)
train_y = np.asarray(TRAIN_LABEL)

# Model weights
# Quadratic function
w = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name="weights")
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="bias")

# Linear Model (Quadratic Function) matmul
prediction = tf.add(tf.matmul(X, w), b)

# MSE
loss = tf.reduce_mean(tf.pow(prediction - Y, 2)) / (2 * TRAIN_DATA.size)
#loss = tf.reduce_mean(tf.square(Y, b))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# Train ==> Decompose optimizer and minimizing...
train = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch):
        print(train_x)
        for (x, y) in zip(train_x, train_y):
            x = np.reshape(x, (1, 2))
            sess.run(train, feed_dict={X: x, Y: y})

        if (epoch - 1) % display_step is 0:
            tmp_x = []
            for idx in range(train_x.size):
                tmp_x.append(np.reshape(train_x[idx], (1, 2)))
                print(tmp_x)
                c = sess.run(loss, feed_dict={X: tmp_x[idx], Y: train_y})
                print("Epoch : ", epoch, " Loss : ", "{:.9f}".format(c), "Weights = ", sess.run(w), " Bias = ",
                      sess.run(b))

    """
    for batch in range(epoch):
        sess.run(train, feed_dict={
            X: train_x,
            Y: train_y
        })
        w_computed = sess.run(w)
        b_computed = sess.run(b)
        print(w_computed, "  ", b_computed)
        # print(w_computed, "  ", b_computed)
    """

