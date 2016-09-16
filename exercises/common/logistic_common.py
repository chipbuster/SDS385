# A file containing common functions for GLMs of wdbc data using
# logistic link function.

import math
import numpy as np
import csv
import sys
from numpy import linalg  as LA
from scipy import linalg  as spla
from scipy import special as spsp
from scipy import stats   as spst

warning_issued = False # Have we issued a warning about possible numerical issues?

def warn_user_numerical():
    """Warn the user that numerical issues may result from value adjustment"""

    global warning_issued

    if not warning_issued:
        print("[WARN]: Adjustments were made to prevent numerical errors. This may cause divergence.")
        print("")
        print("This warning will only appear once in the program even if multiple adjustments are made")
        warning_issued = True

def calc_weight(x_i, B):
    """
    Given parameters beta and data point x_i, calculates the weight parameter

    Both arguments should be column vectors.
    """

    weight = np.asscalar(spsp.expit( np.dot(x_i, B) ))
    if weight == 0:
        weight = 0.1
        warn_user_numerical()
    elif weight == 1:
        weight = 0.9
        warn_user_numerical()

    return weight

def gen_likelihood_function(X,y,m):
    """
    Given problem data, create a function to calculate the likelihood.

    Takes in problem-constant data and returns a function that calculates the
    likelihood at a point.

    X : Data matrix, N x P
    y : response vector, N x 1
    m : trial count vector, N x 1
    """

    # This is a function that's returned
    def likelihood(B):
        result = 0  # value initialization

        weights = [ calc_weight(x,B) for x in X ] #Generate w_i values

        # If the weights are zero or one, we can either terminate the simulation
        # or warn the user and do some adjusting. I choose to do the latter,
        # since the former is achieved by crashing the program.

        # Loop over the individual contribution from each entry and add them
        for i in range(len(weights)):
            result -= np.asscalar(y[i]) * math.log(weights[i])
            result -= np.asscalar(m[i] - y[i]) * math.log(1 - weights[i])

        return result
    return likelihood

def gen_gradient_function(X,y,m):
    """
    Given problem data, create a function to calculate the gradient.

    Takes in problem-constant data and returns a function that calculates the
    gradient at a point.

    X : Data matrix, N x P
    y : response vector, N x 1
    m : trial count vector, N x 1
    """

    def grad(B):
        weights = [ calc_weight(x,B) for x in X ]

        W = np.matrix(np.diag(np.array(weights)))

        gradient = X.T * (y - W * m)
        return gradient

    return grad

def solve_wls(W, X, y):
    """Solve the weighted-least squares problem.

    Solves the weighted least squares problem of the form
       minimize (w.r.t B)     1/2 (y - XB)^T W (x - XB)

    The solution is the same as the solution to the problem

    (X.T * W * X) B = X.T * W * y

    Inputs:
      W : Matrix of P x P, diagonal
      X : Matrix of N x P
      y : vector of P x 1

    Outputs:
      B : Vector of P x 1
    """

    # I could solve this intelligently....but screw it
    # NumPy already has LU-solvers built in.

    A = X.T * W * X
    b = X.T * W * y

    info = spla.lu_factor(A)
    x = spla.lu_solve(A,b)

    return x
