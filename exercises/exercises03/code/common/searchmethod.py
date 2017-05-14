import numpy as np

def backtracking_search(grad_func, obj_func, guess, searchDir):
    """
    Uses backtracking search to find a good step size.

    grad_func:  (function) calculates the gradient
    obj_func:   (function) calculates the objective value
    guess:      (cvector)  current guess (beta)
    searchDir:  (cvector)  direction to search along
    """

    ratio = 0.65    # How much do we decrease the step size each time?
    step = 0.5      # Initial trial value (almost certainly wrong)
    c1 = 1e-6       # Value suggested by book

    while a1 > a2:
        a1 = obj_func(guess + searchDir * step)
        a2 = obj_func(guess) + c1 * step * np.dot(grad_func(guess).T, searchDir)

        step *= ratio

    return step

def grid_search(grad_func, obj_func, guess, searchDir):
    """
    Finds a step size by laying a grid.

    Arguments are the same as for backtracking search.
    """

    step = 1
    ratio = 0.5
    minStep = 1e-10
    grid = [None] * 40

    # Generate the grid
    i = 0
    while step > minStep:
        grid[i] = step
        step = step * ratio
        i += 1

    gridMin = 1e300
    minIndex = -1
    for j in range(i):
        thisMin = obj_func(guess + grid[j] * searchDir)
        if thisMin < gridMin:
            gridMin = thisMin
            minIndex = j

    if minIndex < 0:
        print("Your grid search done fucked up :(")

    return grid[minIndex]
