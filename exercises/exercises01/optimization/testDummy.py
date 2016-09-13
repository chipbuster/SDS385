import csv
import random
import sys
import os
import numpy as np

import steepestdescent
import newtonmethod

def main(filename):
    """Driver for steepest descent code."""

    rawdata = []

    # Read the data in to rawdata
    with open(filename,'r') as ifile:
        r = csv.reader(ifile)
        for row in r:
            rawdata.append(row)

    # First column is always response, rest are predictors

    # Get the data columns from the raw data and convert to matrix
    datalist = [ row[1:] for row in rawdata ]
    predictors = np.array(datalist)
    predictors = predictors.astype(float)

    # Get the responses and convert to 0-1 matrix
    responselist = [ row[0] for row in rawdata ]
    response = np.matrix(responselist).T
    response = response.astype(float)

    # Get the "trial number" for the MLE, in this case, a vectors of 1s
    trials = np.matrix(np.ones(np.shape(response)))

    # Parse filename to get initial guess
    elem = filename.split("=")
    elem[-1] = os.path.splitext(elem[-1])[0]
    
    guesses = [ float(x) for x in elem ]
    guesses = [ x + random.uniform(-x/2, x/2) for x in guesses]

    initGuess = np.matrix(guesses).T

    print(initGuess)

    solution,_ = newtonmethod.solve((predictors, response, trials), initGuess, 1e-2)

    # Since we changed the predictors, need to reverse-transform the solution
    print(solution.T)

if __name__ == '__main__':
    main(sys.argv[1])
