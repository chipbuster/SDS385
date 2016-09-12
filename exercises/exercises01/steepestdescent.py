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
        print("[WARN]: Exponential term capped to avoid overflow. May cause divergence.")
        return math.exp(largest_safe_exponent)
    elif val < smallest_safe_exponent:
        print("[WARN]: Exponential term capped to avoid underflow. May cause divergence.")
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

def zoom(calc_deriv, calc_obj, a, c):
    #Use naive bisection to find trial step
    (alo,ahi) = a
    (c1,c2)  = c

    i = 1
    while True:
        aj = (ahi + alo) / 2
        i += 1

        if calc_obj(aj) > calc_obj(0) + c1 * aj * calc_deriv(0) or calc_deriv(aj) > calc_deriv(alo):
            ahi = aj
        else:
            if abs(calc_deriv(aj)) <= -c2 * calc_deriv(0):
                return aj
            if calc_deriv(aj) * (ahi - alo) >= 0:
                ahi = alo
            alo = aj

        if i > 30:
            print("Breaking from zoom")
            return aj

def line_search(gradFunc, objFunc, guess, amax, c):
    (c1,c2) = c
    i = 1
    found = False

    searchDir = gradFunc(guess)

    # Helper function to calculate scalar derivatives
    def calc_deriv(scal):
        g = gradFunc(guess + searchDir * scal)
        return np.dot(g.T, searchDir)

    # Helper function to calculate objective fn values
    def calc_obj(scal):
        return objFunc(guess + scal * searchDir)

    a = 0.01 * amax
    a_last = 0
    a_min = a # The smallest a value so far

    objZero = calc_obj(0)
    derivZero = calc_deriv(0)

    while True:
        print(i,a)
        if calc_obj(a) > objZero + c1 * a * derivZero or (i > 1 and calc_obj(a) >= calc_obj(a_last)):
            astar = zoom(calc_deriv, calc_obj, (a_last, a), c)
            return astar
        if abs(calc_deriv(a)) <= -c2 * derivZero:
            return a
        if calc_deriv(a) > 0:
            astar = zoom(calc_deriv, calc_obj, (a, a_last), c)
            return astar
        a_last = a
        a = a + (amax - a) * 0.1

        i += 1

        if i > 30 or a - a_last < 1000* sys.float_info.min:
            print("Breaking from line search")
            return a

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
        print("Objective is " + str(llh_op(guess)))
        step = line_search(grad_op, llh_op, guess, 0.5, (0.0001, 0.9))

        guess = guess - grad * step

        delta = abs(llh_op(oldGuess) - llh_op(guess))

        likelihood_record.append(delta)

        print(delta)

    return (guess,likelihood_record)
