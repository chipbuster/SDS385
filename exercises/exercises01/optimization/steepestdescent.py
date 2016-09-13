import math
import numpy as np
import csv
import sys
from numpy import linalg as LA
import pdb

def safe_exp(val):
    """Calculates the "safe exponential" of a value.

    If the computed exponential would result in overflow or underflow,
    it replaces it with a safe value and warns the user."""

    smallest_safe_exponent = math.log(sys.float_info.min) + 3
    largest_safe_exponent = math.log(sys.float_info.max) - 3

    if val > largest_safe_exponent:
        print("[WARN]: Exponential term capped to avoid overflow. May cause divergence.")
        return math.exp(largest_safe_exponent)
    elif val < smallest_safe_exponent:
        print("[WARN]: Exponential term capped to avoid underflow. May cause divergence.")
        return math.exp(smallest_safe_exponent)
    else:
        return math.exp(val)

def calc_weight(x_i,B):
    """Calculates w_i for a given x_i and beta value."""

    return 1 / (1 + safe_exp(-np.dot(x_i, B)))

def calc_likelihood_function(X,y,m):
    """Gives a function to determine the likelihood in the inverse logit method.
    Returns a function which takes in beta and returns the likelihood as a float."""

    def likelihood(B):
        result = 0

        xvecs = [ xs for xs in X ]                        # Length Samples
        weights = [ calc_weight(x,B) for x in xvecs ]

        for i in range(len(xvecs)):
            result -= np.asscalar(y[i]) * math.log(weights[i])
            result -= np.asscalar(m[i] - y[i]) * math.log(1 - weights[i])

        return result
    return likelihood

def calc_grad_function(X,y,m):
    """Calculates the gradient of the inverse logit MLE given the parameters.
    Return is a lambda function which takes a single parameter and returns float."""

    # X is a feature matrix: columns are features, rows are entries. Dim: samples x features
    # y is a response vector: one column of responses                Dim: samples x 1
    # m is a trials vector                                           Dim: samples x 1

    def grad(B):
        # B should be a col vector with length = # features

        xvecs = [ xs for xs in X ]                        # Length Samples
        weights = [ calc_weight(x,B) for x in xvecs ]

        W = np.matrix(np.diag(np.array(weights)))

        gradient = X.T * (y - W * m)
        return gradient

    return grad

def solve(params, initial_guess, converge_step):

    (X,y,m) = params

    # A function which calculates the gradient at a point
    grad_op = calc_grad_function(X,y,m)

    # A function which calculates the likelihood at a point
    llh_op = calc_likelihood_function(X,y,m)

    delta = sys.float_info.max
    guess = initial_guess

    # For storing likelihoods (for tracking convergence)
    likelihood_record = []

    ## Main Steepest Descent Loop
    while delta > converge_step:
        oldGuess = guess

        grad = grad_op(guess)
        step = 0.0001

        guess = guess + grad * step

        delta = abs(llh_op(oldGuess) - llh_op(guess))

        likelihood_record.append(delta)

        print(delta, guess.T)

    return (guess,likelihood_record)
