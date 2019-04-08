import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn.datasets
from testCases1 import *
from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset

def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate *grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate *grads["db"+str(l+1)]
    return parameters


def random_mini_batches(X, Y, mini_batch_size = 64, seed=1):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0 :
        end = m - mini_batch_size*math.floor(m/mini_batch_size)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros_like(parameters["W"]+ str(l+1))
        v["db"+str(l+1)] = np.zeros_like(parameters["b"]+ str(l+1))
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters)
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]
        parameters["dW" + str(l+1)] = parameters["dW" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["db" + str(l+1)] = parameters["db" + str(l+1)] - learning_rate*v["db" + str(l+1)]
    return parameters, v

def initialize_RMSprop(parameters):
    L = len(parameters) // 2
    s = {}
    for l in range(L):
        s["dW" + str(l+1)] = np.zeros_like(parameters["dW" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["db" + str(l+1)])
    return s


def update_parameters_with_RMSprop(parameters, grads, s, t, learning_rate, beta1, epsilon):
    L = len(parameters) // 2
    s_corrected={}
    for l in range(L):
        s["dW" + str(l+1)] = beta1 * s["dW" + str(l+1)] + (1-beta1) * np.power(grads["dW" + str(l+1)],2)
        s["db" + str(l+1)] = beta1 * s["db" + str(l+1)] + (1-beta1) * np.power(grads["db" + str(l+1)],2)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta1, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta1, t))

        parameters["dW" + str(l+1)] = parameters["dW" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["db" + str(l+1)] = parameters["db" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

    return parameters, s


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999, epsilon=1e-8):

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db" + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(beta1,t))

        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.power(grads["dW" + str(l+1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.power(grads["db" + str(l+1)], 2)

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(beta2, t))

        parameters["dW" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / \
                                      (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["db" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / \
                                        (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s


train_X, train_Y = load_dataset()

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64,
          beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10

    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    elif optimizer == "RMSprop":
        s = initialize_RMSprop(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            elif optimizer == "RMSprop":
                parameters, s = update_parameters_with_RMSprop(parameters, grads, s, t, learning_rate, beta2, epsilon)

        if print_cost and i%1000 == 0:
            print("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters














