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

# Add common modules from this project
sys.path.append(os.path.join(os.path.dirname(__file__),'common'))
import logistic_common as lc
import data_common as dc
import plot_common as pc

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
    g -- (scalar) : gamma, the penalty term
    """

    z = np.array(np.shape(B))
    for i in range(np.size(B)):
        xi = x[i]
        z[i] = sign(xi) * max(abs(xi) - g, 0)

    return z

def proximal_gradient(X,y,B):

if __name__ == "__main__":
    main(sys.argv[1])
