#include "sgd_utility.hpp"
#include<ctime>
#include<iostream>

using namespace std;

static inline FLOATING calc_weight(const BetaVec& pred,const BetaVec& estimate){
  /* Calculate w_i for the ith row of the predictor matrix, given a
     current Beta estimate */

  FLOATING exponent = estimate.dot(pred);
  FLOATING w = 1.0 / (1.0 + exp(exponent));

  return w;
}

FLOATING calc_sgd_likelihood(const BetaVec& pred, FLOATING response , const BetaVec& estimate){
  constexpr FLOATING m = 1.0;
  FLOATING w = calc_weight(pred, estimate);
  FLOATING y = response;

  //TODO: Update this for L2 regularization term
  FLOATING contrib = -(y * log(w) + (m - y) * (1 - log(w)));
  return contrib;
}

void calc_sgd_gradient(const BetaVec& pred, FLOATING response, const BetaVec& estimate, BetaVec& gradient){
  static double weight_time = 0.0;
  static double grad_time = 0.0;
  static double extract_time = 0.0;
  clock_t t;

  constexpr FLOATING m = 1.0;

  FLOATING w = calc_weight(pred, estimate);
  FLOATING y = response;

  // gradient = (m * w - y) * x
  // TODO: Update this for L2 regularization term

  FLOATING coefficient = m * w - y;

  t = clock();
  gradient = coefficient * pred;
  t = clock() - t;
  weight_time += double(t) / CLOCKS_PER_SEC;
}
