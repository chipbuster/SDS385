Exercise 04
==============

Contains the code and writeups for Exercise 4, SpeedySGD.

These are in three directories: `code` and `tex`

### tex

Contains the exercise sheets

### code

Contains the code for this exercise. A few notes about these programs:

* Programs can be run on the command line by executing `name-of-program.py wdbc.csv`

* `sgd_minibatch.py` contains the code to test minibatch descent. It is plagued
by issues with the step size search and is generally not recommended.

* `adagrad.py` contains the code for the AdaGrad implementation.

* `adadelta.py` and `rmsprop.py` contain the code for alternatives to Adagrad.
These are generally experimental and should probably not be actually used.

### sgd_c++
 
 This pretty much qualifies for its own repo. See the README.md, hints.md, and
 source code inside this directory for details on how to test it.
