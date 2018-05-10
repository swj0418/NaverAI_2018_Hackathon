import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from pprint import pprint

def getFakeData():
    fake_train_data = np.array([1, 5, 12, 18, 30, 33, 35, 40, 50, 52, 56])
    fake_train_label = np.array([0.1, 0.2, 0.3, .5, .7, .8, 1., 1.1, 1.3, 1.5, 1.8])

    return fake_train_data, fake_train_label

def getPopData():

    HEADER = ['Country', 'Population', 'Yearly_Change', 'Net_Change', 'Density (P / Km2)', 'Land_Area', 'Migrants',
              'Fert_Rate', 'Median_Age', 'Urban_Population', 'World_Share']

    Data: pd.DataFrame = pd.read_csv(
        filepath_or_buffer='F:\\2018_Spring\Programming\Python\Tensorflow\Data\country-pop.csv',
        delimiter='\t', index_col=None, header=None, names=HEADER, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], skiprows=2,
        thousands=',', na_values=('N.A.', np.nan))

    """
    Population Data
    Column 0 : Country Name
    Column 1 : Population
    Column 2 : Yearly Change
    Column 5 : Land Area
    Column 9 : Urban Population
    
    Task 1 : Run a linear regression on a variable 'Population' with a single variable 'Urban_Population'
        *** Train Data : Urban_Population
        *** Train Label: Population
    """
    X_TO_GET = 'World_Share'

    # Replace N.A. Value with 50%
    Data[X_TO_GET].fillna(50, inplace=True)

    temporaryTrainData = []
    for idx in range(Data[X_TO_GET].size):
        temporaryTrainData.append(Data[X_TO_GET].get(key=idx))

    # Replace '%' with a whitespace
    for idx in range(len(temporaryTrainData)):
        temporaryTrainData[idx] = str(temporaryTrainData[idx]).replace('%', '').replace(' ', '')

    train_data = np.asarray(a= temporaryTrainData, dtype=np.float32)
    train_label = np.asarray(a=Data['Population'], dtype=np.float32)

    return train_data, train_label

def singleVariableLinearRegression():
    session = tf.Session()

    train_data, train_label = getPopData()
    # train_data, train_label = getFakeData()

    for idx in range(train_label.size):
        print(train_label[idx], "   ", train_data[idx])

    X: tf.placeholder = tf.placeholder(dtype=np.float32)
    Y: tf.placeholder = tf.placeholder(dtype=np.float32)

    print("Train Data Size  : ", train_data.size)
    print("Train Label Size : ", train_label.size)

    # Parameters
    learning_rate = 0.1
    epoch = 10000
    display_step = 5

    # Model Weights and Bias
    w = tf.Variable(tf.random_uniform([1]), name='weights')
    b = tf.Variable(tf.random_uniform([1]), name='bias')
    # tf.random_uniform([1], -1.0, 1.0)

    # Linear Model (Quadratic Function) matmul
    prediction = tf.add(tf.multiply(w, train_data), b)

    # Cost Function
    lossFunction = tf.reduce_mean(tf.pow(prediction - train_label, 2)) / (2 * train_data.size)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer_02 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5)

    # Train ==> Decompose optimizer and minimizing...
    train = optimizer.minimize(lossFunction)

    np.set_printoptions(suppress=True, formatter={'float_kind': '{:16.3f}'.format}, linewidth=130)

    with session as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epoch):
            sess.run(train)
            if epoch % display_step == 0:
                print("Epoch : ", epoch, "  Weights : ", sess.run(w), "  Bias : ", sess.run(b),
                      " Loss : ", sess.run(lossFunction, feed_dict={X: train_data, Y: train_label}))

        print("Function : Population = ", sess.run(w), " * Urban_Pop(%) + ", sess.run(b))

        t1 = (sess.run(w) * 1.) + sess.run(b)
        t2 = sess.run(w) * 52. + sess.run(b)
        t3 = sess.run(w) * 18. + sess.run(b)

        print(t1)
        print(t2)
        print(t3)

        pyplot.plot(train_data, train_label, 'ro')

        pyplot.plot(train_data, (sess.run(w) * train_data) + sess.run(b), label='Fitted line')
        pyplot.legend()
        pyplot.show()


print("Task 1")
singleVariableLinearRegression()

