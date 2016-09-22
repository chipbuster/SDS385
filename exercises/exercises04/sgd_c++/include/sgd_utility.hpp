#ifndef SGD_UTILITY_H
#define SGD_UTILITY_H

#include<cmath>
#include<cstdlib>
#include<Eigen/Sparse>
#include<Eigen/Dense>

#include "usertypes.hpp"

FLOATING calc_sgd_likelihood(const BetaVec& pred, FLOATING response , const BetaVec& estimate);
void calc_sgd_gradient(const BetaVec& pred, FLOATING response, const BetaVec& estimate, BetaVec& gradient);
#endif
