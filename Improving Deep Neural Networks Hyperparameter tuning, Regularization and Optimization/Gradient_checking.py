import numpy as np
from testCases2 import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J

def backward_propagation(x, theta):
    dtheta = x
    return dtheta

def gradient_check(x, theta, epsilon = 1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus-J_minus) / (2*epsilon)

    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)

    difference = numerator / denominator

    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is worong!")

    return difference



