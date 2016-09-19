import math
import numpy as np
import csv
import sys
import os
import pdb
from numpy import linalg as npla

# Add common modules from this project
sys.path.append(os.path.join(os.path.dirname(__file__),'common'))
import logistic_common as lc
import data_common as dc
import plot_common as pc

from backtrack import backtracking_search
from newtonmethod import gen_hessian_function

def solve(params, initial_guess, converge_step):


    ### VALUE INITIALIZATION

    (X,y,m) = params
    (N,P) = np.shape(X)

    #A function which calculates the gradient at a point
    grad_func = lc.gen_gradient_function(X,y,m)

    # A function which calculates the likelihood at a point
    llh_func = lc.gen_likelihood_function(X,y,m)

    # A function to calculate the Hessian (only used for initialization)
    hess_func = gen_hessian_function(X,y,m)

    delta = sys.float_info.max   # Initial values for change between iteration
    guess = initial_guess
    LLVal = 0             # Dummy likelihood value
    iterct = 0

    # For storing likelihoods (for tracking convergence)
    likelihood_record = []

    ### STEEPEST DESCENT BURN-IN

    # Do a few iterations of steepest descent to burn in Hessian
    B = hess_func(guess)
    while npla.cond(B) > 1e6:
        oldLLVal = LLVal
        oldGuess = guess

        grad = grad_func(guess)
        searchDir = -grad
        step = backtracking_search(grad_func, llh_func, guess, searchDir)

        guess = guess + searchDir * step
        B = hess_func(guess)

        oldGrad = grad

    ### ACTUAL BFGS CODE

    # We now have a suitable B for inversion. Calculate it and update w/ BFGS
    H = B.I
    grad = grad_func(guess) #Also compute gradient so BGFS has a "previous gradient"

    # BFGS Mainloop
    while delta > converge_step:
        # Find search direction and update
        searchDir = -np.dot(H, oldGrad)
        step = backtracking_search(grad_func, llh_func, guess, searchDir)
        guess = guess + searchDir * step

        # Update gradient guess
        grad = grad_func(guess)

        # Update inverse Hessian approximation
        s = (guess - oldGuess)
        y = (grad - oldGrad)
        p = 1 / np.asscalar(np.dot(y.T, s))
        I = np.identity(P)

        # Calculate update matrix (I - p * s * y.T)
        updMat = I - p * s * y.T

        # Update H
        H = updMat * H * updMat + p * s * s.T

        # Calculate current likelihood for convergence determination
        LLVal = llh_func(guess)
        delta = abs( oldLLVal - LLVal )

        # Update recordkeeping values
        oldLLVal = LLVal
        oldGuess = guess
        oldGrad = grad
        likelihood_record.append(LLVal)

        # Update the user and break out if needed
        iterct += 1
        print("Iter: " + str(iterct) + ", objective is: " + str(LLVal)\
              + " Step size is: "  + str(step))
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
    initGuess = np.random.rand(numParams, 1)

    convergeDiff = 1e-4  #Some default value...

    (solution, records) = solve( (X,y,m) , initGuess , convergeDiff )

    # Transform solution back to original space
    solution = W * solution   # Since W is diagonal, inversion is okayish

    # Plot results
    pc.plot_objective(records)

if __name__ == "__main__":
    main(sys.argv[1])
