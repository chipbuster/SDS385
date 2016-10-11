Stochastic Gradient Descent (speed++)
============================================

This is the repository for the speediest gradient descent on this side of the Appalachians.

Okay, not really. But it *is* intended to be a speedy stochastic gradient descent (SGD) which is still easily understandable.

The assignment statement can be found [here](https://github.com/jgscott/SDS385/blob/master/exercises/exercises04.md).

**Please do not copy this code wholesale. I guarantee that the log-likelihood calculations are wrong, and there may be subtle errors in the L2 regularization and the gradient updates.** Instead, use this as a conceptual framework to acelerate your code.

Please also note that the trick of explicit looping with an InnerIterator over the vector may be made obsolete by Eigen 3.3.x.

See Hints.md for some performance things I learned while making this project.

## Prerequisites

In order to build and run this, you must have the following software:

    * C++ compiler (Visual Studio, Clang, GCC, ICC, etc)
    * CMake (version >= 3.2)
    
Note that this build is only tested on Linux and OSX--Windows build suggestions
are in the build instructions, but they are generally untested.

You can get the prerequisites on OSX using [Homebrew](http://brew.sh/) by running

```
brew install eigen cmake
```

Linux users should consult their package manager for details.

Windows users can find CMake installers [on the website](https://cmake.org/download/)
and a C++ compiler in [Visual Studio Express](https://www.visualstudio.com/vs/visual-studio-express/)

## Build Instructions

### Linux / OSX

Make a new directory (not in the source!), and `cd` to it. Then run

```
  cmake <path-to-this-directory> -DCMAKE_BUILD_TYPE=Release 
```

If you wish to use double precision, add `-DUSE_DOUBLES=ON` to the end of
this command.

Then, in the same directory, run `make`. Once finished, you should have the
`runsgd` and `summarize_svmlight` binaries in the build directory.

### Windows

Unfortunately, things are tricky on Windows.

CMake does not allow me to add platform-independent flags for
optimization. You will need to edit the build flags in CMakeLists.txt. Open 
this file in a text editor like Notepad++, then look for the comment  
`# Add CMake debugging and release flags`. On the third line, change
`-Ofast -mavx -march=native -flto -DNDEBUG` to `/Wall /Ox /GL`.

Then activate the CMake GUI. Pick your source and build directories, then 
click on "Configure". When the configuration is done, edit the options by
double clicking on them--the install directory should be where you want
the executables to end up, and the build type should be "Release". Once 
those are set, click on "Configure" again, then click "Generate".

After that, you should be able to find the "ALL_BUILD" project file in the build
directory you specified. Double click on that to open it in Visual Studio,
and build the project.

## Running Instructions

To run either program, do 

```
  <name-of-program> <path-to-svmlight-dir>
```

where `<path-to-svmlight-dir>` is the *directory* containing the `.svm` files.
For example, if I give the path `/tmp/urls`, the program expects to find
`/tmp/urls/Day0.svm`, `/tmp/urls/Day1.svm`, etc.

`summarize_svmlight` is intended to be a sanity test of the parser, and will
give you a summary of the data in the SVMLight files (number of samples,
number of predictors, and percentage of positive responses).

`runsgd` will parse the SVMLight files and do a single iteration of
stochastic gradient descent, timing the iterations.

## Project Parts

This entire project assumes (as a matter of expedience) that m = 1 in all
cases--that is, we only have two possible outcomes. It can easily be
modified to allow for variable m, similar to the Python code.

#### SVMLight Reader

A "simple" C++ library to convert a list of SVMLight files into a set of
Eigen matrices. This project is provided as a simple cpp/hpp, with the
potential to become a library later.

The SVMLight Reader is broken up into two conceptual parts: an entry generator
and a matrix generator. The entry generator, which uses the functions
`parseSVMLightLine`, `readSVMLightFile`, and `readFileList`, takes the
SVMLight files and creates a vector of `Entry` objects. Each `Entry` represents
a single input point.

The matrix generator, in the `genPredictors` function, uses these `Entry` objects
to produce a sparse matrix and a response vector. Further information about this
module can be found in the comment block at the top of `read_svmlight.cpp`

A test driver is created in the `summarize_svmlight` binary. It runs the SVMLight
reader and then gives the user some statistics about the resulting data.

#### SGD

The stochastic gradient descent. Details about the code can be found in the
commenting. Unfortunately, using functions made the initial code quite 
difficult to profile, and I don't have the time to go back and change
the one-megafunction nature of this code now. What can you do :\
