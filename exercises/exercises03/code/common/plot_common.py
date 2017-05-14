#Common plotting routines

import matplotlib.pyplot as plt

def plot_objective(values):
    """Plot a list of values during an optimization"""

    values = list(values) ## Sanity check

    plt.plot(values)
    plt.ylabel("Objective Function Value")
    plt.show()
