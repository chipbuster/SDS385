import inversionsolve as invsolv
import factorsolve as factsolv
import sparsesolve as sparsesolv
import numpy as np
import scipy.sparse as sparse

# TestType: 0 = factor, 1 = inversion
def runTest(rows, cols, testtype, sparsity = 1):
    assert rows > cols, "Rows must be greater than cols"

    W = np.matrix(np.eye(rows))
    X = np.random.rand(rows,cols)
    y = np.random.rand(rows, 1)

    X = X * 100
    y = y * 30

    if testtype == 1 or testtype == "inversion":
        invsolv.solve(X,W,y)
    elif testtype == 0 or testtype == "factor":
        factsolv.solve(X,W,y)
    elif testtype == 2 or testtype == "sparse":
        sparsesolv.solve(X,W,y)
    else:
        print("ERROR: bad test type")
