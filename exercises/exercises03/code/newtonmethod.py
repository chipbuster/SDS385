import numpy as np
from scipy import linalg as spla
from numpy import linalg as npla
import sys
import math
import os
from copy import copy

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

        hess = X.T * D * X

        return hess

    return calc_Hess

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

        grad = grad_func(guess)
        hess = hess_func(guess)

        # Solve for search direction

        searchDir = spla.solve(hess , grad)

        assert npla.norm(hess * searchDir - grad) < 0.1, "Bad solve"

        print(searchDir)

        guess = guess - searchDir * 0.1

        # Calculate current likelihood for convergence determination
        LLVal = llh_func(guess)
        delta = abs( oldLLVal - LLVal )

        likelihood_record.append(LLVal)

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

    (X,y,m) = dc.process_wdbc(csvfile)
    W = dc.prescaling_matrix(X)

    # Scale the predictors
    X = X * W.I

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
