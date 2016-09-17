import numpy as np
from scipy import linalg as spla
from scipy import special as spsp
from numpy import linalg as npla
import sys
import math
import os
from copy import copy
from sklearn.preprocessing import scale
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__),'..','common'))
import logistic_common as lc
import data_common as dc
import plot_common as pc


from scipy.optimize import fmin_cg
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.stats import binom
import scipy.misc as misc

## Note: capital "B" is often used here as a substitute for beta.

## Note 2: This code does not actually calculate the Hessian matrix. Instead, knowing
# that H = X^T D X, we calculate only the D matrix and use the formulae derived in the
# homework to solve the problem.

def gen_hessian_function(X,y,m):
    """Gives a function to calculate the Hessian matrix"""

    N, P = np.shape(X)

    def calc_Hess(B):
        weights = [ lc.calc_weight(x,B) for x in X ]
        oneMinusW = [ 1 - w for w in weights ]

        d_elem = [None] * N
        for j in range(len(weights)):
            d_elem[j] = np.asscalar(m[j]) * weights[j] * oneMinusW[j]


        D = np.diagflat(np.array(d_elem))

        hess = np.matrix(X.T) * np.matrix(D) * np.matrix(X)

        return hess

    return calc_Hess

def gen_D_mat(X,y,m,B):

    (N,P) = np.shape(X)

    weights = [ lc.calc_weight(x,B) for x in X ]
    oneMinusW = [ 1 - w for w in weights ]

    d_elem = [None] * N
    for j in range(len(weights)):
        d_elem[j] = np.asscalar(m[j]) * weights[j] * oneMinusW[j]

    D = np.matrix(np.diagflat(np.array(d_elem)))
    return D

def gen_S_vec(X,y,m,B):

    (N,P) = np.shape(X)

    exp = np.dot(X,B)
    w = spsp.expit(exp)

    W = np.diagflat(w)

    coeffs = np.matrix( W * m - y )
    return coeffs

def solve(params, initial_guess, converge_step):

    (X,y,m) = params

    #A function which calculates the gradient at a point
    grad_func = lc.gen_gradient_function(X,y,m)

    # A function which calculates the likelihood at a point
    llh_func = lc.gen_likelihood_function(X,y,m)

    hess_func = gen_hessian_function(X,y,m)

    delta = sys.float_info.max   # Initial values for change between iteration
    guess = initial_guess
    LLVal = 0             # Dummy likelihood value
    iterct = 0

    # For storing likelihoods (for tracking convergence)
    likelihood_record = []

    ## Main Steepest Descent Lofunc
    while delta > converge_step:
        oldLLVal = LLVal
        oldGuess = guess


#  Alternate solution method
#        D = gen_D_mat(X,y,m,guess)
#        s = gen_S_vec(X,y,m,guess)
#        Z = X * guess - D.I * s
#        searchDir = lc.solve_wls(D,X,Z)


        # Calculate current likelihood for convergence determination
        LLVal = llh_func(guess)
        delta = abs( oldLLVal - LLVal )

        likelihood_record.append(LLVal)

        hess = hess_func(guess)
        grad = grad_func(guess)
        searchDir = spla.solve(hess,grad)

        print("Condition number of Hessian is " + str(npla.cond(hess)))
        print(hess)

        guess = guess + searchDir

        # Update the user and break out if needed
        iterct += 1
        print("Iter: " + str(iterct) + ", objective is " + str(LLVal))
        if iterct > 10000:
            print("Reached 10000 iterations w/o convergence, aborting computation")
            break

    return (guess,likelihood_record)

def main(csvfile):
    # Assume that csvfile points to wdbc.csv
    np.random.seed(5)

    (X1,y,m) = dc.process_wdbc(csvfile)

    # Scale the predictors
    X = scale(X1,axis=1)

    # Generate initial guess
    numParams = np.shape(X)[1]
    initGuess = np.zeros((numParams, 1))

    convergeDiff = 1e-2  #Some default value...

    (solution, records) = solve( (X,y,m) , initGuess , convergeDiff )

    # Transform solution back to original space
    solution = W * solution   # Since W is diagonal, inversion is okayish

    # Plot results
    pc.plot_objective(records)

if __name__ == "__main__":
    main(sys.argv[1])
