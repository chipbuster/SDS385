import os
import sys
import csv
from copy import deepcopy
import random
import math

import numpy as np
import numpy.linalg as npla
import scipy as sp
import scipy.special as spsp  # Learned this one from Amelia
import scipy.stats as spst
import scipy.linalg as spla
from sklearn import preprocessing

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

    grad = X.T @ X @ B - np.matrix(X.T @ y).T

    assert np.shape(grad)[1] == 1, "Gradient is shaped funny!"

    return (2 / np.shape(X)[0]) * grad

def calc_obj(X,y,B,g):
    """Calculate the error of the guess"""
    y = np.matrix(y).T
    myObj = (1 / np.shape(X)[0]) * npla.norm(y - X @ B)**2
    myPen = g * np.sum(np.abs(B))

    return myObj + myPen

def admm(X,y,B, g = 0.1, tol = 1e-5, rho = 7):
    """
    Solve the LASSO problem using proximal gradient.

    X -- (array)   : Predictors, in columns
    y -- (cvector) : Responses
    B -- (cvector) : Initial guess for beta
    g -- (scalar)  : Parameter of lasso
    tol -- (scalar) : Convergence criterion
    """

    deltaerr = 1e10
    lastobj = deltaerr

    mat1 = X.T @ X
    mat1 += rho * np.identity(np.shape(mat1)[0])
    mat2 = X.T @ y

    z = deepcopy(B)
    u = deepcopy(B)

    solveParts = spla.lu_factor(mat1)

    # Run proximal gradient (explanation is in writeup)
    while deltaerr > tol:
        zuDiff = np.matrix(np.reshape(rho * (z - u), np.shape(B)))
        solveFor = mat2 + zuDiff.T
#        print(solveFor.T)
        B = spla.lu_solve(solveParts, solveFor.T)
        z = solve_prox(B + u, g/rho)
        u = u + B - z

        obj = calc_obj(X,y,B,g)
        deltaerr = abs(obj - lastobj)
        lastobj = obj
        print(deltaerr)

    return B

def main(fname1,fname2):
    # Load data
    dataRaw = np.loadtxt(fname1,skiprows=1,delimiter=",")
    responseRaw = np.loadtxt(fname2)
    data = preprocessing.scale(dataRaw)
    response = preprocessing.scale(responseRaw)
    guess = 0.5 * np.ones((np.shape(data)[1],1))

    sol = admm(data,response, guess, 0.1, 1e-7)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
