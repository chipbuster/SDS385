import csv
import random
import sys
import os
import numpy as np


import sgd

def main(filename):

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

    # initGuess = np.matrix([0.709628762541668,-2.34439885373377,-1.06972540358824,
   #           -0.0572196490853194,0.894743019490163, -1.75799494823574,
   #           -1.54356907773596, -0.151108494151544, 1.51847526162427]).T
    # initGuess = np.random.rand(np.shape(initGuess)[0],1) * 0.5 + initGuess

    initGuess = np.matrix(np.zeros( ( 9 , 1 ) ) )

    print(initGuess)

    solution,loglik = sgd.solve((predictors, response, trials), initGuess, 1)

    print(solution.T)
    print()
    print(loglik)

if __name__ == '__main__':
    main(sys.argv[1])
