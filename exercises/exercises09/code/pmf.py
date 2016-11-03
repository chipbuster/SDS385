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

import pdb

def normalize_l2(x):
    """Normalize the vector x in the L2 norm"""
    return x / math.sqrt(np.norm(x))

def softmax(a,c):
    """Compute the elementwise softmax of two vectors"""

    assert(np.shape(a) == np.shape(c))
    return np.multiply(np.sign(a) , np.maximum(np.absolute(a) - c, np.zeros(np.shape(a))))

# Tests for softmax
assert np.all(softmax(np.array([1,2,3]), np.array([-1,0,4])) == np.array([2, 2, 0])),\
       "Your softmax function is broken you tosspot."

def update_with_delta(Xx, maxNorm):
    """Update delta using a tricksy variant on binary search.

    Xx: (cvector) -- The result of multiplying X or X.T by the
                     corresponding vector (either u or v)
    maxNorm:  (scalar)  -- The restriction on the L1 norm of the return value:
                           the L1 norm of the return value must be <=maxNorm

    Return value: x: (vector) -- the vector value, normalized so that L2 == 1
    """

    # silly little helper functions
    def l1(x):
        return npla.norm(x, 1)
    def constvec(c):
        return np.full(np.shape(Xx), c, dtype=np.float_)

    # Try d = 0. If this works, we're home free!
    d = 0
    delta = constvec(d)
    x = softmax(Xx, delta)

    # We're done!
    if l1(x) <= maxNorm:
        return normalize_l2(x)

    # We're not done...need to find a value of d such that l1(x) = maxNorm
    # Start by increasing the value of d until we get past this threshold

    dMax = 1e-6
    while l1(x) > maxNorm:
        dMax *= 2
        delta = constvec(dMax)
        x = softmax(Xx, delta)

    # Now that l1(softmax(Xx,d)) < maxNorm, we can start binary searching
    # Define the boundaries of the search area
    dMin = 1e-6
    d = (dMax + dMin) / 2
    l1x = l1(x)

    while abs(l1x - maxNorm) > 1e-5:
        # Update guess of d
        if l1x > maxNorm:
            dMin = d
        else:
            dMax = d
        d = (dMax + dMin)/2

        delta = constvec(d)
        x = softmax(Xx, delta)
        l1x = l1(x)

    return normalize_l2(x)

def compute_factor(X, v, c1, c2):
    """Compute a single factor of the PMD model with L1 norms

    X: (matrix)    -- The matrix to be factorized
    v: (cvector)   -- Column vector which is the initial guess for v
    c1: (scalar)   -- The L1 bound on u vectors
    c2: (scalar)   -- The L1 bound on v vectors
    """

    assert np.shape(v)[1] == 1,"v is not a column vector"

    v = normalize_l2(v)

    sz_u = np.shape(X)[0]
    sz_v = np.shape(X)[1]

    assert sz_v == np.size(v)

    u = update_with_delta(X @ v)
    v = update_with_delta(X.T @ u)

    delta_u = 1000
    delta_v = 1000

    while delta_u > 1e-5 and delta_v > 1e-5:
        oldU = u
        oldV = v

        u = update_with_delta(X @ v, c1)
        v = update_with_delta(X.T @ u, c2)

        delta_u = np.norm(u - oldU) / sz_u
        delta_u = np.norm(v - oldV) / sz_v

    d = u.T @ X @ v

    return (d,u,v)

def compute_pmf(X, rank, c1, c2):
    """Compute the penalized matrix factorization of X.

    Finds set of U vectors and V vectors such that
      || X - duv.t ||_F is minimized
      ||u||2 <= 1, ||v||2 <= 1
      ||u||1 <= c1, ||v||1 <= c2

    In other words, subject to some sparsity and norm constraints, we should
    have that the sum over i of (d_i * u_i * v_i.T) is close to X.

    X: (np.array)  -- The matrix to be factorized
    rank: (scalar) -- The number of u/v vectors to extract
    c1:   (scalar) -- The L1 bound on u vectors
    c2:   (scalar) -- The L1 bound on v vectors
    """

    X_arr = []
    u_arr = []
    v_arr = []
    d_arr = []

    v_init = np.ones((np.shape(X)[1],1))

    for i in range(rank):
        X_arr.append(X)
        (d,u,v) = compute_factor(X, v_init, c1, c2)
        d_arr.append(d)
        u_arr.append(u)
        v_arr.append(v)

        X -= d * u @ v.T
