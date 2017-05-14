import sys
import os
import math
import csv
import numpy as np

def process_wdbc(filename):
    """Processes the data found in wdbc.csv, returning various terms.

    filename : a string containing the path to wdbc.csv

    Outputs: a (X, y, m) tuple where
        X : an N x P matrix of predictors
        y : an N x 1 vector of responses (converted to 0/1)
        m : an N x 1 vector of trials (in this case, all ones)
    """

    rawdata = []

    # Read the data in to rawdata
    with open(filename,'r') as ifile:
        r = csv.reader(ifile)
        for row in r:
            rawdata.append(row)

    # Get the data columns from the raw data and convert to matrix
    datalist = [ row[2:11] for row in rawdata ]
    predictors = np.array(datalist)       # We'll convert this to matrix later
    predictors = predictors.astype(float)

    # Attach column of ones for intercept term
    onecol = np.matrix(np.ones(np.shape(predictors)[0])).T
    predictors = np.matrix(np.concatenate((predictors, onecol),axis=1))

    # Get response vector and convert it to 0-1
    responselist = [ 1 if row[1] == "M" else 0 for row in rawdata ]
    response = np.matrix(responselist).T

    # Get trial count vector
    trials = np.matrix(np.ones(np.shape(response)))

    #            X         y        m
    return (predictors, response, trials)

def prescaling_matrix(X):
    """Scales column of a matrix.

    The predictors of a matrix may be horribly off-scale from each other
    (e.g. same distance measured in cm vs km), which can cause numerical
    issues.

    As a crude fix, scale all the columns by their mean value so that
    the predictors have similar orders of magnitude.

    X : an N x P matrix of predictors

    Outputs:
        W : A P x P diagonal matrix with scaling factors.
            X' = W * X to get scaled predictor matrix
    """
    condWeights = np.average(X,axis=0)   # Average down the columns
    W = np.matrix(np.diagflat(condWeights))

    return W
