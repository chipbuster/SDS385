import os
import sys
import csv
from copy import copy
import random
import math

import numpy as np
import scipy as sp
import scipy.special as spsp  # Learned this one from Amelia
import scipy.stats as spst
import numpy.linalg as npla

# Add common modules from this project
sys.path.append(os.path.join(os.path.dirname(__file__),'common'))
import logistic_common as lc
import data_common as dc
import plot_common as pc

import pdb

#temporary dummy function
def calc_gamma():
    return 0.1

def sign(x):
    """Return +1 if x >= 0, and -1 otherwise"""
    return 1 if x >= 0 else -1

def solve_prox(B, g):
    """Solve the proximal operator of P(x) = t |x|

    Inputs:
    B -- (cvector) : The vector where the proximal operator should be evaluated
    g -- (scalar)  : gamma, the penalty term
    """

    z = np.zeros(np.shape(B))

    for i in range(np.size(B)):
        xi = B[i]
        z[i] = sign(xi) * max(abs(xi) - g, 0)

    return z

def calc_gradient(X,y,B):
    """Calculate the gradient of f(B) = (XB - y)^T(XB - y)."""

    grad = B.T @ X.T @ X
    grad -= y.T @ X
    grad = grad.T

    assert np.shape(grad)[1] == 1, "Gradient is shaped funny!"

    return 2 * grad


def proximal_gradient(X,y,B, g, tol):
    """
    Solve the LASSO problem using proximal gradient.

    X -- (array)   : Predictors, in columns
    y -- (cvector) : Responses
    B -- (cvector) : Initial guess for beta
    g -- (scalar)  : Parameter of lasso
    tol -- (scalar) : Convergence criterion
    """

    err = npla.norm(X @ B - y)

    # Run proximal gradient (explanation is in writeup)
    while err > tol:
        grad = calc_gradient(X,y,B)
        u = B - g * grad
        B = solve_prox(u,g)

        err = npla.norm(X @ B - y)
        print(err)

    return B

def main(fname1,fname2):
    # Load data
    data = np.loadtxt(fname1,skiprows=1,delimiter=",")
    response = np.loadtxt(fname2)
    guess = np.zeros((np.shape(data)[1],1))

    sol = proximal_gradient(data,response, guess, 0.1, 0.01)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
