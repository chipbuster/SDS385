## This solves (X^T W X) B = X^T W y by using matrix inversion on the LHS
## Notation: I treat this as an Ax = b problem, so
##   A = X^T W X
##   b = X^T W y
##   x = beta

import numpy as np
import scipy as sp
import random

def gen_A(X, W):
    return X.T * W * X

def gen_b(X, W, y):
    return X.T * W * y

def solve(X, W, y):
    A = gen_A(X,W)
    b = gen_b(X,W,y)

    solution = A.I * b
    return solution
