
# Math libraries
import numpy as np
import scipy as sp
import math
import numpy.linalg as npla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from scipy.sparse.linalg import spsolve
from numpy import inf
import random
import time

#Imaging libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Data import libraries
import reader

# Debugging libraries
import code
import pdb

tol = 1e-13
globallambda = 0.2

def pred_shape(mat):
    """If mat is a data matrix, return the shape of its predictor."""
    return (np.shape(mat)[1], 1)

def direct_solve(D,b,lam):
    "Solve the smoothing problem with a sparse direct solver. Variables same as assignment."

    (m,n) = np.shape(D)
    ident = spsp.identity(n)
    A = spsp.csc_matrix(lam * D.T @ D + ident)

    lu = spspla.splu(A)
    solution = lu.solve(b)

    solution.reshape((np.size(solution), 1))

    return solution

def gs_solve(D,b,lam):
    "Solve the smoothing problem with Gauss-Seidel iteration. Variables same as assignment."

    (m,n) = np.shape(D)
    ident = spsp.identity(n)
    A = lam * D.T @ D + ident

    # the matrix a is our target, we want to solve lx = b. gauss-seidel splits
    # this into two components: lower and strictly-upper triangular
    L = spsp.tril(A, format='csc')
    U = A - L

    assert(spspla.norm( A - (L + U), inf) < 1e-3 )

    Linv = spspla.inv(L)

    guess = spsp.csc_matrix(np.zeros(pred_shape(A)))
    err = 10

    while err > tol:
        temp = (b - U @ guess)
        guess = Linv @ temp
        result = A @ guess
        err = npla.norm(result - b, 2)
 #       print(err)

    return guess


def jacobi_solve(D,b,lam):
    "Solve the smoothing problem with Jacobi iteration. Variables same as assignment."

    (m,n) = np.shape(D)
    ident = spsp.identity(n)
    A = lam * D.T @ D + ident

    # the matrix a is our target, we want to solve lx = b. jacobi iteration splits
    # this into two components: diagonal and off-diagonal.
    diagMat = spsp.spdiags(A.diagonal(), 0, n,n,format="csc") #Need CSC to invert
    offDiagMat = A - diagMat

    # Invert the diagonal matrix elementwise (scipy inverse uses LU solve)
    invDiagMat = spsp.spdiags(1.0/A.diagonal(), 0, n,n,format="csc")

    #Sanity checks: - is the diagonal matrix times its inverse the identity?
    #               - If we join the diag and off-diag elements, do we get A?
    #               - is the diagonal of offDiagMat exactly zero?
    assert(spspla.norm(diagMat @ invDiagMat - ident, inf) < 1e-3)
    assert(spspla.norm( A - (diagMat + offDiagMat), inf) < 1e-3 )
    assert(npla.norm(offDiagMat.diagonal(),2) < 1e-3)

    guess = spsp.csc_matrix(np.zeros(pred_shape(A)))
    err = 10

    while err > tol:
        guess = invDiagMat @ (b - offDiagMat @ guess)
        result = A @ guess
        err = npla.norm(result - b, 2)
#        print(err)

    return guess

def main(fname):
    # Read in matrices, set up graphs
    y = reader.read_file(fname)
    edgeSz = int(math.sqrt(np.size(y)))
    D = reader.gen_D_matrix(edgeSz)

    # Solve directly, with timer, average of 3
    print("Testing Direct Solver")
    t = time.perf_counter()
    direct = direct_solve(D,y,globallambda)
    direct = direct_solve(D,y,globallambda)
    direct = direct_solve(D,y,globallambda)
    direct_time = (time.perf_counter() - t) / 3.0

    print("Testing Gauss-Seidel")
    t = time.perf_counter()
    gs = gs_solve(D,y,globallambda)
    gs = gs_solve(D,y,globallambda)
    gs = gs_solve(D,y,globallambda)
#    gs = direct
    gs_time = (time.perf_counter() - t) / 3.0

    print("Testing Jacobi Iteration")
    t = time.perf_counter()
    jacobi = jacobi_solve(D,y,globallambda)
    jacobi = jacobi_solve(D,y,globallambda)
    jacobi = jacobi_solve(D,y,globallambda)
#    jacobi = direct
    jacobi_time = (time.perf_counter() - t) / 3.0

    print(" --- \n")

    # Solution sanity check
    (m,n) = np.shape(D)
    ident = spsp.identity(n)
    A = globallambda * D.T @ D + ident

    res_direct = direct - y
    res_gs     = gs - y
    res_jacobi = jacobi - y

    print("Direct Solve averaged " + str(direct_time) +  " per solve.")
    print("Gauss Seidel averaged " + str(gs_time) + " per solve.")
    print("Jacobi Iteration averaged " + str(jacobi_time) + " per solve.")

    print("")

    print("2-norm of direct solution residual is " + str(npla.norm(res_direct)))
    print("2-norm of Gauss-Seidel solution residual is " + str(npla.norm(res_gs)))
    print("2-norm of Jacobi solution residual is " + str(npla.norm(res_jacobi)))

    print("")

    print("Largest elementwise difference between solutions:\n---")
    print("\t Direct vs Gauss-Seidel: " + str(npla.norm(direct - gs,inf)))
    print("\t Direct vs Jacobi: " + str(npla.norm(direct - jacobi,inf)))
    print("\t Jacobi vs Gauss-Seidel: " + str(npla.norm(jacobi - gs,inf)))

    # Reshape our solutions into images for imshow to display
    orig = np.reshape(y, (edgeSz,edgeSz))
    gs_smooth = np.reshape(gs, (edgeSz,edgeSz))
    direct_smooth = np.reshape(direct, (edgeSz,edgeSz))
    jacobi_smooth = np.reshape(jacobi, (edgeSz,edgeSz))

    # Plot results
    plt.figure(1)
    plt.suptitle("Smoothing Results")

    plt.subplot(221)
    plt.imshow(orig)
    plt.title("Original Image")
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(gs_smooth)
    plt.title("Gauss-Seidel")
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(direct_smooth)
    plt.title("Direct")
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(jacobi_smooth)
    plt.title("Jacobi")
    plt.colorbar()


    #Plot difference from original
    plt.figure(2)
    plt.suptitle("Difference from Original Image")

    plt.subplot(221)
    plt.imshow(orig - orig)
    plt.title("Original Image - Original Image")
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(gs_smooth - orig)
    plt.title("Gauss-Seidel")
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(direct_smooth - orig)
    plt.title("Direct")
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(jacobi_smooth - orig)
    plt.title("Jacobi")
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    main('fmri_z.csv')
