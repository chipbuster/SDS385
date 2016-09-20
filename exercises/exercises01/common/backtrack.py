import numpy as np

def backtracking_search(grad_func, obj_func, guess, searchDir):
    """
    Uses backtracking search to find a good step size.

    grad_func:  (function) calculates the gradient
    obj_func:   (function) calculates the objective value
    guess:      (cvector)  current guess (beta)
    searchDir:  (cvector)  direction to search along
    """

    ratio = 0.65 # How much do we decrease the step size each time?
    step = 0.5     # Initial trial value (almost certainly wrong)
    c1 = 1e-6    # Value suggested by book

    # Dummy values for iteration
    a1 = 10
    a2 = 1

    while a1 > a2:
        a1 = obj_func(guess + searchDir * step)
        a2 = obj_func(guess) + c1 * step * np.dot(grad_func(guess).T, searchDir)

        print(a1,a2)

        step *= ratio

    return step
