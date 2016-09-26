#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>
#include<cstring>

#include "usertypes.hpp"
#include "read_svmlight.hpp"
#include "tinydir.h"

#ifndef NDEBUG
#include <gperftools/profiler.h>
#endif

/* Runs a single iteration of SGD through the entire datase (picking each
 point exactly once). Does not return a value, guess is modified in-place */
void sgd_iteration(PredictMat& pred, ResponseVec& r, DenseVec& guess,
                   FLOATING regCoeff = 1e-4,
                   FLOATING masterStep = 1e-1){
  /* Args:

     pred : the sparse matrix of predictors
     r    : the vector of responses (0-1)
     guess: the guess for gradient descent
     regCoeff : The L2 regularization coefficient
     masterStep: the master step size for Adagrad */

  constexpr FLOATING adagradEpsilon = 1e-7;

  int nPred = pred.cols();
  BetaVec gradient = BetaVec(nPred);
  DenseVec agWeights = DenseVec::Random(nPred); //Adagrad weights
  DenseVec epsVec = DenseVec::Constant(nPred, adagradEpsilon);

  #ifndef NDEBUG
  ProfilerStart("gperftools.out");
  #endif

  constexpr FLOATING m = 1.0;

  clock_t t;
  double t1 = 0;
  double t2 = 0;
  double t3 = 0;

  for(int i = 0; i < 9000 /*pred.rows()*/; i++){
    //Calculate the gradient

    t = clock();
    BetaVec predSamp = pred.row(i);
    FLOATING exponent = predSamp.dot(guess);
    FLOATING w = 1.0 / (1.0 + exp(-exponent));
    FLOATING y = r(i);
    FLOATING coefficient = m * w - y;
    t = clock() - t;
    t1 += (double)t / CLOCKS_PER_SEC;

    t = clock();
    gradient = coefficient * predSamp;
    gradient.eval();
    t = clock() - t;
    t2 += (double)t / CLOCKS_PER_SEC;

    // Update the guess by the AdaGrad rules
    t = clock();
    // guess += gradient * 0.0001;
    for(BetaVec::InnerIterator it(gradient); it; ++it){
      int q = it.index();
      guess(q) += it.value() * 0.0001;
    }
    guess.eval();
    t = clock() - t;
    t3 += (double)t / CLOCKS_PER_SEC;

    // Update the AdaGrad weights
    // agWeights += gradient.cwiseProduct(gradient);

    cout << "Index is " << i << endl;
  }

  cout << t1 << "  " << t2 << "  " << t3 << endl;

  #ifndef NDEBUG
  ProfilerStop();
  #endif

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
  DenseVec guess = DenseVec::Constant(predictors.cols(), 0.1);
  sgd_iteration(predictors, responses, guess);
}
