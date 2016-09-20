Stochastic Gradient Descent (speed++)
============================================

This is the repository for the speediest gradient descent on this side of the Appalachians.

Okay, not really. But it *is* intended to be a speedy stochastic gradient descent (SGD) which is still easily understandable.

The assignment statement can be found [here](https://github.com/jgscott/SDS385/blob/master/exercises/exercises04.md).

## Prerequisites

In order to build and run this, you must have the following software:

    * Unix system (sorry Windows users)
    * C++ compiler
    * Eigen3
    * CMake (version >= 3.2)
    * Boost (?)

You can get these on OSX using [Homebrew](http://brew.sh/) by running

```
brew install eigen cmake
```

Linux users should consult their package manager for details.

## Build Instructions

Coming soon to a README near you!

## Project Parts

#### SVMLight Reader

A "simple" C++ library to convert a list of SVMLight files into a set of
Eigen matrices. This project is provided as a simple cpp/hpp, with the
potential to become a library later.
