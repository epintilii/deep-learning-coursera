import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
#plt.show()

shape_X = X.shape
shape_Y = Y.shape

m = Y.shape[1]

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)



