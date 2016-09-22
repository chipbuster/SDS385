#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>

#include "usertypes.hpp"
#include "read_svmlight.hpp"
#include "sgd_utility.hpp"

/* Runs a single iteration of SGD through the entire datase (picking each
 point exactly once). Does not return a value, guess is modified in-place */
void sgd_iteration(PredictMat& pred, ResponseVec& r, BetaVec& guess,
                   FLOATING convergence){
  constexpr FLOATING adagradEpsilon = 1e-7;

  int nPred = guess.rows();
  BetaVec gradient = BetaVec(nPred);
  BetaVec agWeights; //Adagrad weights, stored as a vector
  BetaVec epsVec = BetaVec::Const(nPred, adagradEpsilon);
  for(int i = 0; i < pred.rows(); i++){
    
  }

}
