## This solves (X^T W X) B = X^T W y by using matrix factorization on the LHS
## Notation: I treat this as an Ax = b problem, so
##   A = X^T W X
##   b = X^T W y
##   x = beta

import numpy as np
import scipy as sp
from scipy import linalg
import random

def solve(A,b):

    LUFact = linalg.lu_factor(A)
    solution = linalg.lu_solve(LUFact, b)

    return solution
