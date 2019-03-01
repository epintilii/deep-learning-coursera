import math
import numpy as np
import time

def basic_sigmoid(x):
    s = 1 / (1+math.exp(-x))
    return s

x = np.array([1, 2, 3])

def sigmoid(x):
    s = 1./(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s *(1-s)
    return ds

print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

def image2vector(image):
    v = image.reshape(image.size, 1)
    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])



def normalizeRows(x):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x

x = np.array([
    [0, 3, 4],
    [2, 6, 4]])



def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


x = np.array([[9, 2, 5, 0, 0],
              [7, 5, 0, 0, 0]])



#Vectorization
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]


### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic  = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] + x2[i]
toc = time.process_time()

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i]*x2[j]

toc = time.process_time()

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3, len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x[j]

toc = time.process_time()


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]


### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1, x2)
toc =time.process_time()

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.mutiply(x1,x2)
toc = time.process_time()

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
mul = np.dot(W,x1)
toc = time.process_time()

def L1(yhat, y):
    loss = np.sum(np.abs((y-yhat)))
    return loss

def L2(yhat, y):
    loss = np.sum(np.square(yhat - y))
    return loss

