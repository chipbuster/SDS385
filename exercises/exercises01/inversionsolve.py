## This solves (X^T W X) B = X^T W y by using matrix inversion on the LHS
## Notation: I treat this as an Ax = b problem, so
##   A = X^T W X
##   b = X^T W y
##   x = beta

import numpy as np
import scipy as sp
import random

def solve(A,b):

    solution = A.I * b
    return solution
