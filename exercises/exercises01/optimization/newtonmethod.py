import numpy as np
from scipy import linalg
import sys
import math

## Note: capital "B" is often used here as a substitute for beta.

## Note 2: This code does not actually calculate the Hessian matrix. Instead, knowing
# that H = X^T D X, we calculate only the D matrix and use the formulae derived in the
# homework to solve the problem.

def linsolve(A,b):
    lu = linalg.lu_factor(A)
    return linalg.lu_solve(lu,b)

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

def calc_s_vector(X,y,m):
    """Gives a function to calculate the vector s, where s_i = y_i - m_i w_i"""

    # X is a feature matrix: columns are features, rows are entries. Dim: samples x features
    # y is a response vector: one column of responses                Dim: samples x 1
    # m is a trials vector                                           Dim: samples x 1

    def calc_s(B):
        # B should be a col vector with length = # features

        xvecs = [ xs for xs in X ]                        # Length Samples
        weights = [ calc_weight(x,B) for x in xvecs ]
        w = np.matrix(weights).T                          #Transpose to get a column vector

        s = np.matrix(y - np.multiply(m,w))       # yi - mi * wi

        return s

    return calc_s

def calc_D_matrix(X, y, m):
    """Gives a function to calculate the diagonal matrix D, where D(i,i) = m_i w_i (1 - w_i)."""

    def calc_D(B):
        xvecs = [ xs for xs in X ]                        # Length Samples
        weights = [ calc_weight(x,B) for x in xvecs ]
        w = np.matrix(weights).T                          # Transpose to make column vector

        d = np.multiply(m, w)
        oneMinusW = np.ones(np.shape(w)) - w
        d = np.multiply(m, oneMinusW)

        D = np.diagflat(d)     #Generate diagonal array with d as its diagonal

        return D

    return calc_D

# This one does a direct calculation since it needs so many parameters
def calc_Z_vector(D, s, X, B0):
    """Calculate the value of Z given the input parameters"""

    part1 = X * B0
    part2 = linsolve(D,s)   # D.I * s --> D x = s

    return part1 - part2

def solve(params, initial_guess, converge_criteria):

    (X,y,m) = params

    # A function which calculates the likelihood at a point
    llh_op = calc_likelihood_function(X,y,m)

    # A function which calculates the s-vector
    get_s = calc_s_vector(X,y,m)

    # A function which calculates the D-matrix
    get_D = calc_D_matrix(X,y,m)

    delta = sys.float_info.max
    guess = initial_guess

    # For storing likelihoods (for tracking convergence)
    likelihood_record = []

    ## Main Steepest Descent Loop
    while delta > converge_criteria:
        oldGuess = guess

        s = get_s(guess)
        D = get_D(guess)
        Z = calc_Z_vector(D, s, X, guess)

        # We seek to minimize 1/2 (z - X * B).T D (z - X * B)

        A = X.T * D * X
        b = X.T * D * Z

        step = linsolve(A,b)

        guess = guess - step

        delta = abs(llh_op(oldGuess) - llh_op(guess))

        likelihood_record.append(delta)

        print(delta)

    return (guess,likelihood_record)
