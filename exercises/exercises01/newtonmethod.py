import numpy as np
from scipy import linalg
import sys

def linsolve(A,b):
    lu = linalg.lu_factor(A)
    return linalg.lu_solve(lu,b)

def gen_gradient(z, X, W):
    pass

def gen_hessian(z, X, W):
    pass

def gen_likelihood(z, X, W):
    pass

def solve(params, initial_guess, converge_criteria):

    # A function which calculates the gradient at a point
    grad_op = calc_grad_function(params)

    # A function which calculates the likelihood at a point
    llh_op = gen_gradient(params)

    # A function which calculates the Hessian at a point
    hess_op = gen_hessian(params)

    delta = sys.float_info.max
    guess = initial_guess

    # For storing likelihoods (for tracking convergence)
    likelihood_record = []

    ## Main Steepest Descent Loop
    while delta > converge_criteria:
        oldGuess = guess

        grad = grad_op(guess)
        hess = hess_op(guess)

        step = linsolve(hess, grad)

        guess = guess - step

        delta = abs(llh_op(oldGuess) - llh_op(guess))

        likelihood_record.append(delta)

        print(delta)

    return (guess,likelihood_record)
