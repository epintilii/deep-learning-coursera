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

