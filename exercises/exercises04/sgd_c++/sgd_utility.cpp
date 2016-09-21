#include "sgd_utility.hpp"

FLOATING calc_weight(PredictMat& pred, int i, BetaVec& estimate){
  /* Calculate w_i for the ith row of the predictor matrix, given a
     current Beta estimate */

  FLOATING exponent = pred.row(i).dot(estimate);
  FLOATING w = 1.0 / (1.0 + exp(exponent));
  return w;
}

BetaVec calc_sgd_gradient(PredictMat& pred, ResponseVec& response, int sampleNum, BetaVec& estimate){
  /* Calculates the gradient contribution of a single point. Takes in the whole
     matrix and response, selects the relevant elements, and performs calc.*/

  constexpr FLOATING m = 1.0;
            FLOATING w = calc_weight(pred, sampleNum, estimate);
            FLOATING y = response(sampleNum);

  // gradient = (m * w - y) * x
  // TODO: Update this for L2 regularization term

  FLOATING coefficient = m * w - y;

  BetaVec gradient = coefficient * BetaVec(pred.row(i));
  return gradient;
}

FLOATING calc_sgd_likelihood(PredictMat& pred, ResponseVec& response, int sampleNum, BetaVec& estimate){

  constexpr FLOATING m = 1.0;
            FLOATING w = calc_weight(pred, sampleNum, estimate);
            FLOATING y = response(sampleNum);

  //TODO: Update this for L2 regularization term
  FLOATING contrib = -(y * log(w) + (m - y) * (1 - log(w)));
  return contrib;
}
