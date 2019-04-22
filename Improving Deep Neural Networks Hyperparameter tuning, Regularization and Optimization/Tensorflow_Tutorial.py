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










