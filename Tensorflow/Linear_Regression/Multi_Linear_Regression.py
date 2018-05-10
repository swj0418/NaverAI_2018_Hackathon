import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from matplotlib import pyplot

def getStudentData(X: []):
    filePath: str = "F:\\2018_Spring\Programming\Python\Tensorflow\Data\student-mat.csv"

    train_data = []
    train_label = []

    totalDataFrame: pd.DataFrame = pd.read_csv(filepath_or_buffer=filePath, sep=';', header='infer')
    totalDataFrameIndex: [] = totalDataFrame.keys()

    # From totalDataFrame, extract three columns passed on to the function
    for idx in range(len(X)):
        dataFrame: pd.DataFrame = totalDataFrame[totalDataFrameIndex[X[idx]]]
        train_data.append(np.asarray(dataFrame))

    # From totalDataFrame, extract labels which are grade points. USE FINAL GRADE FOR NOW. (Single y)
    y_columnNumber = 31
    labelSeries: pd.DataFrame = totalDataFrame[totalDataFrameIndex[y_columnNumber]]
    train_label = np.asarray(labelSeries)

    return train_data, train_label


def Normalize(train_data: np.array):
    normalized = []

    # Normalization done with all three series composed together
    norm = preprocessing.normalize(train_data, norm='l2')

    for idx in range(len(train_data)):
        normalized.append(np.asarray(norm[idx], dtype=np.float32))

    return normalized


def predict(train_data, weights):
    """
    :param train_data:  (Data Length, N-features)
    :param weights:     {N-featurs, 1)
    :return:
    """

    train_data = np.transpose(train_data)
    return np.dot(train_data, weights)

def cost_function(train_data, train_label, weights):
    """
    :param train_data: (Data Length, N-feature)
    :param train_label:  (Data Length, 1)
    :param weights:  (N-Features, 1)
    :return: 1-Dimensional matrix of predictions
    """

    N = len(train_label)

    predictions = predict(train_data, weights)

    sqError = (predictions - train_label) ** 2

    return np.float32(1.0 / (2*N) * sqError.sum())

# Gradient Descent Optimizer native implementation.
def update_weights(features, targets, weights, lr):
    '''
    Features:(200, 3)
    Targets: (200, 1)
    Weights:(3, 1)
    '''
    predictions = predict(features, weights)

    #Extract our features
    x1 = features[:,0]
    x2 = features[:,1]
    x3 = features[:,2]

    # Use matrix cross product (*) to simultaneously
    # calculate the derivative for each weight
    d_w1 = -x1*(targets - predictions)
    d_w2 = -x2*(targets - predictions)
    d_w3 = -x3*(targets - predictions)

    # Multiply the mean derivative by the learning rate
    # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
    weights[0][0] -= (lr * np.mean(d_w1))
    weights[1][0] -= (lr * np.mean(d_w2))
    weights[2][0] -= (lr * np.mean(d_w3))

    return weights


def Multivariate_Regression():
    print("Multivariate")

    # Parameters
    learningRate: np.float32 = 0.001
    epoch: np.int32 = 10000
    printInterval: np.int32 = 1000

    #                  2      6      7
    # Prepare data : AGE, M_EDU, F_EDU
    train_data, train_label = getStudentData([2, 6, 7])

    # Normalize
    train_data = Normalize(train_data)
    train_data = np.asarray(train_data)
   #train_data = np.transpose(train_data)

    train_label = np.asarray(train_label)

    # Defined tf.Session()
    session: tf.Session() = tf.Session()

    # MODEL
    # Grade = W1 * AGE + W2 * M_EDU * W3 * F_EDU
    # Define Weight, and Bias
    weights = tf.Variable(tf.random_uniform([train_label.size, 1], -1.0, 1.0))

    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

    # Placeholders
    X = tf.placeholder(tf.float32, [3, train_label.size], name="X")
    Y = tf.placeholder(tf.float32, [train_label.size], name="Y")

    # Cost Function
    y_ = tf.matmul(X, weights)
    costFunction = tf.reduce_mean(tf.square(y_ - Y) + b)

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)

    # Train Function
    train = optimizer.minimize(costFunction)

    # Session Beginning
    with session as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(epoch):
            session.run(train, feed_dict={X: train_data, Y: train_label})
            if epoch % printInterval == 0:

                print("Epoch : ", epoch,
                      " Cost : ", session.run(costFunction, feed_dict={X: train_data, Y: train_label}))

        print("Computation Complete=================================================================================")
        print("Function ::: ", "Grade = ", session.run(tf.matmul(np.asmatrix(train_data[0]), weights)), "*X_1 + ",
              session.run(tf.matmul(np.asmatrix(train_data[1]), weights)), "*X_2 + ",
              session.run(tf.matmul(np.asmatrix(train_data[1]), weights)), "*X_3 + ", session.run(b))


Multivariate_Regression()