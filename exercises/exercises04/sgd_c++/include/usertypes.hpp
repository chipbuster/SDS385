// you need to also include Eigen/Sparse and Eigen/Dense when using this file
#ifndef USERTYPES_H
#define USERTYPES_H

#ifdef USE_DOUBLES
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> PredictMat;
typedef Eigen::VectorXd ResponseVec;
typedef Eigen::VectorXd DenseVec;
typedef Eigen::SparseVector<double> BetaVec;
typedef double FLOATING;
#else
typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int> PredictMat;
typedef Eigen::VectorXf ResponseVec;
typedef Eigen::VectorXf DenseVec;
typedef Eigen::SparseVector<float> BetaVec;
typedef float FLOATING;
#endif


#endif
