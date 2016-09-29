Stochastic Gradient Descent Performance Hints
==========================================================

While there are a lot of micro-performance tips, there's two big themes that you
should make sure you're always following:

* Don't update anything you don't need to
* Access your sparse data efficiently

Let's take a look at these a little more closely:

### Caveat Programmer

When I say that things are "expensive", I mean that they're costly to do, but even
a relatively expensive operation might take only a millisecond. Of course, if
you need to do this once per sample, you're now looking at
2.4 million * 0.001 second = 2400 seconds or 40 minutes of time spent on that
alone.

The point here is that if you need to do something costly just once or twice in 
your code (as part of initialization, for example), it probably won't make a
measurable difference. If you need to do it repeatedly, in the part of your
code that's run the most often, you might want to reconsider how your program
is structured.

## Update At The Last Possible Moment

#### First Calculation

The most straightforward way to calculate the update for beta looks something 
like this:

```c++
    SparseVector samp = data.inner(sampnum);
    
    //Calculate weights and gradient

    gradient = someDenseVector;  //Calculated earlier and expressed as a dense vec
    adaGradWeights += gradient.cwiseSquare();
    adaGradWeights = (adaGradWeights + adagradEpsilon).cwiseSqrt();
    adaGradWeights = adaGradWeights.cwiseInverse();
    beta -= gradient.cwiseProduct(adaGradWeights);
```

This will work, but it'll be really slow.

The reason is because your average gradient has 2.4 million elements, but only
115 of them are non-zero. The above code explicitly calculates every single 
element, so your computer will spend 99.995% of its time square rooting zeros.

#### Sparse Vectors Fail Hard

You'd think we could solve the problem by making the gradient vector sparse.
Then Eigen would know that when we do `gradient.cwiseSquare()`, it's only
supposed to square the nonzero elements of the gradient, and maybe we could do
something similar for the adaGradWeights vector, and everything would
magically go fast!

Unfortunately, Eigen's support for sparse vectors isn't stellar, and in some
cases it reverts to dense operation. What this means for us is that this
strategy won't work for speeding up the SGD calculations--at some point, Eigen
will revert back to dense vector operations and we'll lose all of our
performance. So while this may be something you'll be able to do in future
versions of Eigen, when sparse support improves, it's not a strategy that
will work at the present.

### Explicit Sparseness To The Rescue

Instead, we can use sparse iterators (see 
[this link](http://gallery.rcpp.org/articles/sparse-iterators/)
if you don't know what these are) to explicitly calculate only the sparse 
elements: instead of giving Eigen a sparse vector and hoping that it knows
how to do things sparsely, we explicitly tell Eigen which elements to calculate.

```c++
SparseVector dataSample = data.inner(sampnum);
    
// calculate weights

SparseVector::InnerIterator it(dataSample);
for(  ; it; ++it){
    int j = it.index();

    //Calculate gradient(j)
    adaGradWeights(j) += gradient(j) * gradient(j);
    adaGradWeights(j) = 1.0 / sqrt(adaGradWeights(j) + adaGradEps);
    beta(j) -= adaGradWeights(j) * gradient(j);
    
    }
```

Compare this code with the one in the section "First Calculation" and convince
yourself that they really are the same thing. Note that we're now explicitly
telling Eigen to only compute the nonzero elements, so it doesn't really
matter if the gradient vector is sparse or not, because we only update
nonzeros either way.

(In fact, it's possible to entirely eliminate the gradient vector altogether
and use a scalar value for the gradient calculations, but I'll let you
figure out how to do it.)

This version of the code will take you 80-90% of the way to a speedy SGD.
Unfortunately, there's one issue which could still potentially bring your
program to an excruciating crawl...

#### How to deal with regularization

If you're eagle-eyed, you'll notice that so far, we haven't dealt with
regularization at all. Regularization shrinks every element of the gradient
at every iteration, so if you calculate it explicitly, you have to modify every
nonzero element of beta at every iteration. It seems that we have no choice
but to do dense updates and lose performance.

Or do we? 

Let's do a thought experiment. Suppose you have a book which needs
to have all its words crossed out by the time you give it to your friend
one week from now (your friend is a bit eccentric). You give it to a bookstore
which specializes in this task.

**Does it matter if the bookstore crosses out the words a few at a time over the 
course of the week or crosses it all out at once?**

Hopefully you'll agree that the answer is "no," as long as a) they finish by
the end of the week, and b) you don't need to look at the book in the meantime.

We find ourselves in a similar situation with updating the beta. We *technically*
need to update every element of beta at every iteration...but if we don't look
at, read, modify, or use that element in any form, what do we care if the update
actually happens every iteration or not, as long as we have the right answer at the end?

This leads us to the idea of *lazy updating*. If you look at the SGD code, you'll
notice that we only need to access the elements of beta when we're updating them
with gradient calculations. We never use those values otherwise. This means
that we can defer updates for them between element accesses for the gradient.

To do this, we keep a vector that tells us on which iteration we last updated
the L2 penalty. Then, we calculate the accumulated penalty before updating
the element.

The total pseudocode is now

```c++
SparseVector samp = data.inner(sampnum);

// calculate weights

SparseVector::InnerIterator it(dataSample);
for(  ; it; ++it){
    int j = it.index();
    
    int numSkipped = iterNum - lastUpdate[j]
    calculate_l2_penalty_here(numSkipped, otherData); // Use your imagination!
    lastUpdate[j] = iterNum;
    
    //Calculate gradient(j)
    adaGradWeights(j) += gradient(j) * gradient(j);
    adaGradWeights(j) = 1.0 / sqrt(adaGradWeights(j) + adaGradEps);
    beta(j) -= adaGradWeights(j) * gradient(j);
}
```

This setup works quite well for getting speedy speedy SGD.

## Accessing Sparse Data

#### Accessing Sparse Matrices

In Eigen, it's possible to extract the rows and columns of a sparse matrix
with operators: `spmat.row(i)` extracts the ith row of a sparse matrix,
and `spmat.col(i)` extracts its columns.

However, if the matrix is column-major, extracting rows will be *incredibly*
slow, since Eigen will need to do a search for every single element of the
resulting vector.

To be safe, I'd recommend just using an 
[InnerIterator](http://gallery.rcpp.org/articles/sparse-iterators/)
and forgetting about explicitly accessing rows and columns.

However, if you really want to play with rows and columns, make sure
you only extract rows from row-major matrices and columns from
column-major matrices. You can control whether the matrix is
row or column major with typedefs, e.g.:

```c++
    typedef SparseMatrix<double, Eigen::RowMajor, int> SpMatRowMaj;
    typedef SparseMatrix<double, Eigen::ColMajor, int> SpMatColMaj;
```

#### Accessing Sparse Vectors

NB: This section is only relevant if you decide to experiment with storing
your gradient/beta as a sparse vector.

Similarly, it is very expensive to access an explicit element of a sparse
vector. Doing `spvec.coeffRef(i)` forces Eigen to search through the
entire vector to find the element you requested. Instead, if you need to
modify all the nonzeros of sparse data, use the `.valueRef()` method of an
InnerIterator, e.g.

```c++
    SparseVector::InnerIterator it(mySparseVector);
    //Advance it a couple of times here
    
    int j = it.index();
    
    it.coeffRef(j) = newValue; //Slow: forces search
    it.valueRef() = newValue; // Very fast!
```

This avoids expensive searches on the sparse data.

### Other Tricks

#### Fast Inverse Square Root

If your other functions are well optimized enough, you might eventually find
that a majority of your time is taken up in calls to the standard math
library: `log`, `exp`, and `sqrt`. While we can't do much about the first two,
it's possible to speed the last one up: in particular, we don't actually 
need to calculate the square root *per se*, we need to calculate the
reciprocal of the square root.

Fortunately, we can calculate that directly with 
[Carmack's function](https://en.wikipedia.org/wiki/Fast_inverse_square_root), 
which can drastically speed things up. Unfortunately, the actual speedup
will vary drastically with the compiler, operating system, and actual
code you write<sup>1</sup>, so this is an optimization that you're better of 
testing after your code is complete.

#### Compiler Flags

The default compiler flags for Rcpp use the second optimization level
and no vectorization. You may be able to slightly decrease your code's
runtime by using the `-O3` optimization level and an appropriate
vectorization flag (either `-msse2`, `-mavx`, or `maxv2`). However, I have
no idea how to do this and have never actually tested it. If you're compiling
raw C++, just use the `-O3` flag.

<sup>1</sup> I've also suspected it depends on the phase of the moon, the location of 
Jupiter relative to certain constellations, and whether the dread god 
Cthulhu has been having a good day or not.
