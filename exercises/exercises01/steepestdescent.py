import math
import numpy as np
import csv
import sys
from numpy import linalg as LA
import pdb

bump = 0.00000001 # A tiny bump for some values that really should not be zero
smallest_safe_exponent = math.log(sys.float_info.min) + 3
largest_safe_exponent = math.log(sys.float_info.max) - 3

def safe_exp(val):
    """Calculates the "safe exponential" of a value. If the computed exponential would
    be too large, it replaces it with a safe value."""

    if val > largest_safe_exponent:
        return math.exp(largest_safe_exponent)
    elif val < smallest_safe_exponent:
        return math.exp(smallest_safe_exponent)
    else:
        return math.exp(val)

def calc_likelihood_function(X,y,m):
    """Gives a function to determine the likelihood in the inverse logit method.
    Returns a function which takes in beta and returns the likelihood as a float."""

    def likelihood(B):
        result = 0

        xvecs = [ xs for xs in X ]                        # Length Samples
        exponents = [ np.dot(x,B) for x in xvecs ]
        expterms = [ safe_exp(z) for z in exponents ]
        weights = [ 1.0 / (1.0 + e) for e in expterms ]

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
        exponents = [ np.dot(x,B) for x in xvecs ]
        expterms = [ safe_exp(z) for z in exponents ]
        weights = [ 1.0 / (1.0 + e) for e in expterms ]

        W = np.matrix(np.diag(np.array(weights)))

        gradient = X.T * (y - W * m)
        return gradient

    return grad

def steepest_descent(params, initial_guess, converge_step):

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
        step = 0.001

        guess = guess - grad * step

        delta = abs(llh_op(oldGuess) - llh_op(guess))

        likelihood_record.append(delta)

        print(delta)

    return (guess,likelihood_record)

def main(filename):
    """Driver for steepest descent code."""

    rawdata = []

    # Read the data in to rawdata
    with open(filename,'r') as ifile:
        r = csv.reader(ifile)
        for row in r:
            rawdata.append(row)

    # Get the data columns from the raw data and convert to matrix
    datalist = [ row[2:11] for row in rawdata ]
    predictors = np.array(datalist)
    predictors = predictors.astype(float)

    # Attach column of ones for intercept term
    onecol = np.matrix(np.ones(np.shape(predictors)[0])).T

    predictors = np.matrix(np.concatenate((predictors, onecol),axis=1))

    # Get the responses and convert to 0-1 matrix
    responselist = [ 1 if row[1] == "M" else 0 for row in rawdata ]
    response = np.matrix(responselist).T

    # Get the "trial number" for the MLE, in this case, a vectors of 1s
    trials = np.matrix(np.ones(np.shape(response)))

    # Recondition matrix to have order of magnitude ~1
    condWeights = np.average(predictors,axis=0)
    W = np.matrix(np.diagflat(condWeights))

    predictors = predictors * W.I

    initGuess = np.matrix(np.ones((np.shape(predictors)[1],1))) * 0.001

    solution,_ = steepest_descent((predictors, response, trials), initGuess, 0.01)

    # Since we changed the predictors, need to reverse-transform the solution
    solution = W * solution

    print(solution)
