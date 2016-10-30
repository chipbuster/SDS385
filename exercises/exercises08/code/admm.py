import os
import sys
import numpy as np
import scipy as sp
import math
import random
import time
import numpy.linalg as npla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from scipy.sparse.linalg import spsolve
from numpy import inf
from copy import deepcopy

#Imaging libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import reader

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

    gvec = np.ones(np.shape(B)) * g

    z = np.multiply(np.sign(B) , np.maximum(np.absolute(B) - gvec, np.zeros(np.shape(gvec))))

    z.reshape((np.size(z), 1))

    return z

def calc_gradient(X,y,B):
    """Calculate the gradient of f(B) = (XB - y)^T(XB - y)."""

    grad = X.T @ X @ B - np.matrix(X.T @ y).T

    assert np.shape(grad)[1] == 1, "Gradient is shaped funny!"

    return (2 / np.shape(X)[0]) * grad

def calc_obj(F,y,B,g):
    """Calculate the objective function"""

    obj = 0.5 * npla.norm(y - B)**2 + g * npla.norm(F @ B)

    return obj

def admm(F, y,B, g = 0.1, tol = 1e-5, rho = 7):
    """
    Solve a modified LASSO problem using proximal gradient.

    F -- (matrix)  : Transform to apply
    y -- (cvector) : Responses
    B -- (cvector) : Initial guess for beta
    g -- (scalar)  : Parameter of lasso
    tol -- (scalar) : Convergence criterion

    Solves min(x) : (1/2) || Ax - y||_2^2 + g||r||1
    Subject to Fx - r = 0
    """

    FF = F.T @ F
    ident = spsp.identity(np.shape(FF)[0])

    deltaerr = 1e10
    lastobj = deltaerr

    B = spsp.csc_matrix(B)

    z = deepcopy(B)
    u = deepcopy(B)

    # Generate the LU factorization of the matrix we need to solve for
    mat1 = spsp.csc_matrix(rho * FF + ident)
    solveParts = spspla.splu(mat1) #Generate a SuperLU object

    # Precompute vector part
    mat2 = y

    # Run proximal gradient (explanation is in writeup)
    while deltaerr > tol:
        zuDiff = z - u
        solveFor = mat2 + rho * F.T @ zuDiff
        B = solveParts.solve(solveFor)
        z = solve_prox(F @ B + u, g/rho)
        u = u + F @ B - z

        obj = calc_obj(F,y,B,g)
        deltaerr = abs(obj - lastobj)
        lastobj = obj
        print(deltaerr)

    return B

def main(fname):

    lam = 30

    # Load data
    y = reader.read_file(fname)
    edgeSz = int(math.sqrt(np.size(y)))
    D = reader.gen_D_matrix(edgeSz)

    guess = np.zeros((np.shape(D)[0],1))

    sol = admm(D,y, guess, lam, 1e-5)

    (m,n) = np.shape(D)
    ident = spsp.identity(n)
    A = lam * D.T @ D + ident

    orig = np.reshape(y, (edgeSz,edgeSz))
    admm_smooth = np.reshape(y, (edgeSz,edgeSz))

    plt.figure(1)
    plt.suptitle("Smoothing Results")

    plt.subplot(121)
    plt.imshow(orig)
    plt.title("Original Image")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(admm_smooth)
    plt.title("ADMM")
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main('fmri_z.csv')
