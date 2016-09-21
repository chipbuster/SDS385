#include "sgd_utility.hpp"

FLOATING calc_weight(PredictMat& pred, int i, BetaVec& estimate){
  /* Calculate w_i for the ith row of the predictor matrix, given a
     current Beta estimate */

  FLOATING exponent = pred.row(i).dot(estimate);
  FLOATING w = 1.0 / (1.0 + exp(exponent));
  return w;
}
