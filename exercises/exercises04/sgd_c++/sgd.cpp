#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>
#include<cstring>
#include<ctime>

#include "usertypes.hpp"
#include "read_svmlight.hpp"
#include "tinydir.h"

#ifndef NDEBUG
#include <gperftools/profiler.h>
#endif

using namespace std;

static inline FLOATING weight(BetaVec& guess, BetaVec& samp){ return samp.dot(guess);}
static inline FLOATING weight(DenseVec& guess, BetaVec& samp){ return samp.dot(guess);}

/* Runs a single iteration of SGD through the entire datase (picking each
 point exactly once). Does not return a value, guess is modified in-place */
void sgd_iteration(PredictMat& pred, ResponseVec& r, BetaVec& sparseGuess,
                   FLOATING regCoeff = 1e-4,
                   FLOATING masterStepSize = 1e-1){
  /* Args:

     pred : the sparse matrix of predictors
     r    : the vector of responses (0-1)
     guess: the guess for gradient descent
     regCoeff : The L2 regularization coefficient
     masterStep: the master step size for Adagrad */

  constexpr FLOATING adagradEpsilon = 1e-7;
  constexpr FLOATING m = 1.0;

  int nPred = pred.cols();
  int nSamp = pred.rows();
  DenseVec agWeights = DenseVec::Random(nPred); //Adagrad weights
  DenseVec guess = DenseVec(sparseGuess);

  static double time1 = 0.0;
  static double time2 = 0.0;
  clock_t t;

  for(int test = 0; test < 20; test++){
    t = clock();
  for(int i = 0; i < pred.rows(); i++){
    //Calculate the gradient
    BetaVec predSamp = pred.row(i);

    // P1: sparse dot dense, in code
    FLOATING psi0 = predSamp.dot(guess);
    // P2: sparse dot sparse, in code
    // FLOATING psi0 = predSamp.dot(sparseGuess);
    // P3: sparse dot dense, in func
    //FLOATING psi0 = weight(guess, predSamp);
    // P4: sparse dot sparse, in code: /
    //FLOATING psi0 = weight(sparseGuess,predSamp);

    FLOATING epsi = exp(psi0);
    FLOATING yhat = m / (1.0 + epsi);

    FLOATING delta = r(i) - yhat;

    for(BetaVec::InnerIterator it(predSamp); it; ++it){
      int k = it.index();
      int j = k;

      // Calculate gradient for this element
      FLOATING elem_gradient = -delta * it.value();

      // Update weights for Adagrad
      agWeights(j) += elem_gradient * elem_gradient;
      FLOATING h = sqrt(agWeights(i) + adagradEpsilon);

      // Scale element
      FLOATING scaleFactor = masterStepSize / h;

      guess(j) -= scaleFactor * elem_gradient;
      //it.valueRef() -= scaleFactor * elem_gradient;
    }

    //   cout << "Index is " << i << " and nnz is " << guess.nonZeros() <<  endl;
  }
  cout << "Time taken for core iters: " << static_cast<double>(clock() - t)/CLOCKS_PER_SEC << "s" << endl;
  }
}

int main(int argc,char** argv){
  tinydir_dir dir;
  tinydir_file file;
  string dirname = string(argv[1]);
  tinydir_open(&dir, dirname.c_str());

  vector<string> filenames;

  while (dir.has_next){
    tinydir_readfile(&dir, &file);
    tinydir_next(&dir);
    if(file.is_dir) continue; //Not interested in directories

    size_t len = strlen(file.name);
    if(len < 4                 ||
       file.name[len-1] != 'm' ||
       file.name[len-2] != 'v' ||
       file.name[len-3] != 's' ||
       file.name[len-4] != '.'){
      continue; //This file does not end in ".svm"
    }
    filenames.push_back(dirname + string(file.name));
  }

  tinydir_close(&dir);

  // Parse the files
  vector<Entry> results = readFileList(filenames);
  std::pair<ResponseVec, PredictMat> out = genPredictors(results);
  cout << "Matrices generated. " << endl;
  ResponseVec responses = out.first;
  PredictMat predictors = out.second;

  // Do SGD
  BetaVec guess = DenseVec::Constant(predictors.cols(), 0.0).sparseView();
  sgd_iteration(predictors, responses, guess);
}
