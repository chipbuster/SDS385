Stochastic Gradient Descent (speed++)
============================================

This is the repository for the speediest gradient descent on this side of the Appalachians.

Okay, not really. But it *is* intended to be a speedy stochastic gradient descent (SGD) which is still easily understandable.

The assignment statement can be found [here](https://github.com/jgscott/SDS385/blob/master/exercises/exercises04.md).

**IF YOU SEE A TYPE YOU DON'T UNDERSTAND, CHECK TO SEE IF IT IS IN `usertypes.h` FIRST!**

## Prerequisites

In order to build and run this, you must have the following software:

    * C++ compiler
    * Eigen3
    * CMake (version >= 3.2)
    
Note that this build is only tested on Linux and OSX--it may be possible
to use the Visual Studio generators on Windows, but it is completely
untested.

You can get the prerequisites on OSX using [Homebrew](http://brew.sh/) by running

```
brew install eigen cmake
```

Linux users should consult their package manager for details.

## Build Instructions

Coming soon to a README near you!

## Project Parts

This entire project assumes (as a matter of expedience) that m = 1 in all
cases--that is, we only have two possible outcomes.

#### SVMLight Reader

A "simple" C++ library to convert a list of SVMLight files into a set of
Eigen matrices. This project is provided as a simple cpp/hpp, with the
potential to become a library later.

#### S
