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

class Samples:
    """A class to store and select samples from a data frame.

    Stores predictors and responses as matrices, then selects elements
    without replacement. To do this, it keeps a set of valid index
    numbers. If the valid set has values, then one is selected and removed,
    and used to index into the predictors and responses. If there are no
    valid indices, an end-of-epoch is marked and the valid index set is
    restored.
    """
    def __init__(self,X,y,m):
        """Initialize samples from data.

        X: N x P matrix of predictors
        y: N x 1 vector of responses
        m: N x 1 vector of trial counts
        """

        assert np.shape(X)[0] == np.shape(y)[0] and \
               np.shape(X)[0] == np.shape(m)[0] ,\
            "Number of rows in the input data does not match"

        assert np.shape(y)[1] == 1,\
            "y is not a column vector (column count is not 1)"

        assert np.shape(m)[1] == 1,\
            "y is not a column vector (column count is not 1)"


        self.predictors = X
        self.responses = y
        self.trials = m

        # Available samples, by index. When we "remove" a sample, we remove
        # its index from availIndices.
        self.availIndices = set(list(range(np.shape(X)[0])))

        self.epochs = 1           # Epoch count, we start in the first epoch

    def new_epoch(self):
        """Restores the predictors and responses by copying out of originals."""
        self.availIndices = set(list(range(np.shape(self.predictors)[0])))
        self.epochs += 1

    def get_sample(self):
        """Select samples from internal pool.

        Samples without replacement from internal pool of points. If internal
        pool runs out, restores the internal pool by copying from originals.

        Returns tuple (x,y), where y is scalar response and x is row-vector
        of predictors.
        """

        # If no indices available, refill pool and mark end of epoch
        if not self.availIndices:
            self.new_epoch()

        ## Choose an index and remove it from the pool
        index = random.choice(tuple(self.availIndices))
        self.availIndices.remove(index)

        # Use index to select sample
        sampleX = self.predictors[index]
        sampley = self.responses[index]
        samplem = self.trials[index]

        return (np.matrix(sampleX).T, sampley, samplem)

def calc_sgd_step(B, x, y, m):
    """Calculates the next step in SGD.

    B: (cvector) -- Current guess for beta
    x: (cvector) -- Predictors of a single sample point
    y: (scalar)  -- Response of sample point
    m: (scalar)  -- Number of trials on this sample point, m_i
    """

    ## Using expit, we can calculate all this in one function! Magic!
    w = spsp.expit( np.dot(x.T,B) )

    gradient = np.asscalar( m * w - y ) * x

    return gradient

def solve(params, initial_guess, converge_step):
    """Calculates optimization problem with stochastic descent."""

    (X,y,m) = params

    llh_func = lc.gen_likelihood_function(X,y,m) #Function to calculate likelihood

    samplePoints = Samples(X,y,m) # Create class for sampling points

    delta = sys.float_info.max   # Initial values for change between iteration
    guess = initial_guess
    LLVal = 0             # Dummy likelihood value
    iterct = 0


    likelihood_record = []

    while delta > converge_step:
        oldLLVal = LLVal
        oldGuess = guess

        (xSamp, ySamp, mSamp) = samplePoints.get_sample()

        searchDir = calc_sgd_step(guess, xSamp, ySamp, mSamp)
        step = 0.001

        guess = guess - searchDir * step

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
    initGuess = np.random.rand(numParams, 1)

    convergeDiff = 1e-4  #Some default value...

    (solution, records) = solve( (X,y,m) , initGuess , convergeDiff )

    # Transform solution back to original space
    solution = W * solution   # Since W is diagonal, inversion is okayish

    # Plot results
    pc.plot_objective(records)

if __name__ == "__main__":
    main(sys.argv[1])
