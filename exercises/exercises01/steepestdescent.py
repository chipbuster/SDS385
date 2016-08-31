import math
import numpy as np
import csv
import sys
from numpy import linalg as LA

bump = 0.00000001 # A tiny bump for some values that really should not be zero

def calc_likelihood_function(X,y,m):
    """Gives a function to determine the likelihood in the inverse logit method.
    Returns a function which takes in beta and returns the likelihood as a float."""

    def likelihood(B):
        result = 0

        xs  = [ xi for xi in X.T ]
        exp = [ math.exp(-np.dot(xi,B)) + bump for xi in xs ] #In case xi dot B = 0, add bump
        w   = [ 1.0 / (1 + et ) for et in exp ]

        for i in range(np.shape(X)[1]):
            result += np.asscalar(y[i]) * math.log(1 + exp[i])
            result -= np.asscalar(m[i] - y[i]) * math.log(exp[i] / (1 + exp[i]))

        return result
    return likelihood

def calc_grad_function(X,y,m):
    """Calculates the gradient of the inverse logit MLE given the parameters.
    Return is a lambda function which takes a single parameter and returns float."""

    def grad(B):
        gradient = np.matrix(np.zeros(np.shape(B)))

        xs  = [ xi.T for xi in X.T ]
        exp = [ math.exp(-np.dot(B.T,xi)) + bump for xi in xs ] #In case xi dot B = 0, add bump
        w   = [ 1.0 / (1 + et ) for et in exp ]

        for i in range(np.shape(X)[1]):
            update = np.asscalar(y[i]) * w[i] * exp[i] * xs[i]
            gradient += update
            gradient -= np.asscalar(m[i] - y[i]) * w[i] * w[i] * exp[i] * xs[i]

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
        gradient = grad_op(guess)
        prev_guess = guess

        # Search along gradient direction using MAGIC
        step = 1 if delta == sys.float_info.max else delta * 100
        guess = guess + gradient * step

        prev_guess_val = llh_op(prev_guess)
        curr_guess_val = llh_op(guess)

        likelihood_record.append(curr_guess_val)

        delta = abs(prev_guess_val - curr_guess_val)

    return (guess, likelihood_record)

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

    solution,_ = steepest_descent((predictors, response, trials), trials * 0.001, 0.00000001)

    print(solution)
