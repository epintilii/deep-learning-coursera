import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
# print(tf.__version__)
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

y_hat = tf.constant(36, name="y_hat")
y = tf.constant(39, name="y")

loss = tf.Variable((y-y_hat)**2, name="loss")
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)

sess = tf.Session()
print(sess.run(c))

x = tf.placeholder(tf.int64, name = "x")
print(sess.run(2 * x, feed_dict={x:3}))
sess.close()

def linear_function():
    np.random.seed(1)
    x = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W, x), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x:z})
    return result

def cost(logits, labels):
    z = tf.placeholder(tf.float32, name="x")
    y = tf.placeholder(tf.float32, name="y")

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z:logits, y:labels})
    return cost

def one_hot_matrix(labels, C):
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    return one_hot

def ones(shape):
    ones = tf.ones(shape)

    with tf.Session() as sess:
        ones=sess.run(ones)
    return ones


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)



def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())

    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters