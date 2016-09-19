Exercise 03
==============

Contains the code and writeups for Exercise 3, Line Search + BFGS.

These are in two directories: `code` and `tex`

### tex

Contains the tex sources and pdfs for the writeup and exercise sheets

### code

Contains the code for this exercise. A few notes about these programs:

* `steepestdescent.py` was essentially copied verbatim from ex01, but modified
to use the line search method from backtrack.py instead of using a fixed step size.

* `newtonmethod.py` is not intended to be executed--it's only in here so I can
use the hessian functions. Perhaps this should be refactored into `common` at some point in the future.

* Both `bfgs.py` and `steepestdescent.py` can be run with the following command:`python3 <pythonfile.py> wdbc.csv`
