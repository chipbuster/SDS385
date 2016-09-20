import math
import numpy as np
import csv
import sys
import os
import pdb

# Add common modules from this project
sys.path.append(os.path.join(os.path.dirname(__file__),'common'))
import logistic_common as lc
import data_common as dc
import plot_common as pc

from backtrack import backtracking_search

def solve(params, initial_guess, converge_step):

    (X,y,m) = params

    #A function which calculates the gradient at a point
    grad_func = lc.gen_gradient_function(X,y,m)

    # A function which calculates the likelihood at a point
    llh_func = lc.gen_likelihood_function(X,y,m)

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
        searchDir = -grad
        step = backtracking_search(grad_func, llh_func, guess, searchDir)

        guess = guess + searchDir * step

        # Calculate current likelihood for convergence determination
        LLVal = llh_func(guess)
        delta = abs( oldLLVal - LLVal )

        likelihood_record.append(LLVal)

        # Update the user and break out if needed
        iterct += 1
        print("Iter: " + str(iterct) + ", objective is " + str(LLVal) + " step size: " + str(step)) 
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
