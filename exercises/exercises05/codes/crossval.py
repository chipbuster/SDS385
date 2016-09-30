import numpy as np
from numpy.linalg import norm
import math
from sklearn.linear_model import Lasso

def crossValLasso(X, y, lambdaval , fraction = 10):
    """
    Cross validate a Lasso model

    X = data (as numpy array, predictors in columns)
    y = responses (as numpy array)
    lambdaval = float, value for lambda
    fraction = an integer. 1/fraction is the fraction
               of the hold out data, e.g. fraction=10
               is 10-fold cross validation

    return: a list of mean squared error resulting from
            each crossval trial, with len = fraction
    """

    (N,P) = np.shape(X)

    ## Generate boundaries for the crossval. To be honest, this is probably
    # wrong but I can't bring myself to care enough
    boundaries = np.linspace(0, N, fraction).tolist()
    boundaries = [ int(x) for x in boundaries ]

    errors = []

    for j in range(len(boundaries) - 1):
        toDelete = range(boundaries[j], boundaries[j+1])
        first = toDelete[0]
        last = toDelete[-1]
        trainX = np.delete(X, toDelete, axis=0)
        trainy = np.delete(y, toDelete, axis=0)
        testX = X[first:last,:]
        testy = y[first:last]

        thisLasso = Lasso(alpha = lambdaval)
        thisLasso.fit(trainX,trainy)

        predicted = thisLasso.predict(testX)

        thisError = norm(testy - predicted)**2 / np.size(testy)

        errors.append(thisError)

    return errors
