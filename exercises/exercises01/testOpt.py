import csv
import numpy as np

import steepestdescent
#import newtonmethod

def main(filename):
    """Driver for steepest descent code."""

    rawdata = []

    # Read the data in to rawdata
    with open(filename,'r') as ifile:
        r = csv.reader(ifile)
        for row in r:
            rawdata.append(row)

    # Get the data columns from the raw data and convert to matrix
    datalist = [ row[2:11] for row in rawdata ]
    predictors = np.array(datalist)
    predictors = predictors.astype(float)

    # Attach column of ones for intercept term
    onecol = np.matrix(np.ones(np.shape(predictors)[0])).T

    predictors = np.matrix(np.concatenate((predictors, onecol),axis=1))

    # Get the responses and convert to 0-1 matrix
    responselist = [ 1 if row[1] == "M" else 0 for row in rawdata ]
    response = np.matrix(responselist).T

    # Get the "trial number" for the MLE, in this case, a vectors of 1s
    trials = np.matrix(np.ones(np.shape(response)))

    # Recondition matrix to have order of magnitude ~1
    condWeights = np.average(predictors,axis=0)
    W = np.matrix(np.diagflat(condWeights))

    predictors = predictors * W.I

    initGuess = np.matrix(np.random.rand(np.shape(predictors)[1],1)) * 0.001

    solution,_ = steepestdescent.solve((predictors, response, trials), initGuess, 1e-16)

    # Since we changed the predictors, need to reverse-transform the solution
    solution = W * solution

    print(solution)

if __name__ == '__main__':
    main('wdbc.csv')
