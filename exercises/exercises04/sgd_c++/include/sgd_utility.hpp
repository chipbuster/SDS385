#ifndef SGD_UTILITY_H
#define SGD_UTILITY_H

#include<cmath>
#include<cstdlib>
#include<Eigen/Sparse>
#include<Eigen/Dense>

#include "usertypes.hpp"

FLOATING calc_weight(PredictMat& pred, int sampleNum, BetaVec& estimate);
FLOATING calc_sgd_gradient(PredictMat& pred, ResponseVec& response, int sampleNum, BetaVec& estimate);
FLOATING calc_sgd_likelihood(PredictMat& pred, ResponseVec& response, int sampleNum, BetaVec& estimate);

#endif
