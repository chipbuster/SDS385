import inversionsolve as invsolv
import factorsolve as factsolv
import sparsesolve as sparsesolv
import numpy as np
import scipy.sparse as sparse

# TestType: 0 = factor, 1 = inversion
def genTest(rows, cols, sparsity = 1):
    assert rows > cols, "Rows must be greater than cols"

    W = sparse.identity(rows)
    X = sparse.rand(rows,cols,sparsity)
    y = sparse.rand(rows,1   ,sparsity)

    X = X * 100
    y = y * 30

    A = X.T * W * X
    b = X.T * W * y

    A_dense = A.todense()
    b_dense = b.todense()

    return (A,b,A_dense, b_dense)
