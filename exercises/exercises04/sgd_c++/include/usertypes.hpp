// you need to also include Eigen/Sparse and Eigen/Dense when using this file

#ifdef USE_DOUBLES
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> PredictMat;
typedef Eigen::VectorXd ResponseVec;
typedef Eigen::VectorXd BetaVec;
typedef double FLOATING;
#else
typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int> PredictMat;
typedef Eigen::VectorXf ResponseVec;
typedef Eigen::VectorXf BetaVec;
typedef float FLOATING;
#endif
