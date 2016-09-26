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

#define BEGIN_TIME() t = clock()
#define END_TIME(var) t = clock() - t; var += (double)t / CLOCKS_PER_SEC

//Using Carmack's magic, adapted for doubles
#ifdef USE_DOUBLES
static inline double invSqrt( const double& x )
{
  double y = x;
  double xhalf = ( double )0.5 * y;
  long long i = *( long long* )( &y );
  i = 0x5fe6ec85e7de30daLL - ( i >> 1 );//LL suffix for (long long) type for GCC
  y = *( double* )( &i );
  y = y * ( ( double )1.5 - xhalf * y * y );
  y = y * ( ( double )1.5 - xhalf * y * y ); //For better accuracy

  return y;
}
#else
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	constexpr float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
#endif

using namespace std;

/* Runs a single iteration of SGD through the entire datase (picking each
 point exactly once). Does not return a value, guess is modified in-place */
DenseVec sgd_iteration(PredictMat& pred, ResponseVec& r, DenseVec& guess,
                   FLOATING regCoeff = 1e-4,
                   FLOATING masterStepSize = 1e-1){
  /* Args:

     pred : the sparse matrix of predictors
     r    : the vector of responses (0-1)
     guess: the guess for gradient descent
     regCoeff : The L2 regularization coefficient
     masterStep: the master step size for Adagrad */

  //Compile-time constants--baked into code
  constexpr FLOATING adagradEpsilon = 1e-7;
  constexpr FLOATING m = 1.0;
  constexpr FLOATING movingAverageRatio = 0.1;

  //Timing variables
  clock_t t;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;

  int nPred = pred.cols();
  int nSamp = pred.rows();

  //Adagrad weights--the zeroeth element is for the intercept term, so all
  //accesses for elements need to be offset by one.
  DenseVec agWeights = DenseVec::Random(nPred).cwiseAbs();

  //Tracker for the value of the objective function (here, nll_avg)
  DenseVec objTracker = DenseVec::Zero(nPred);
  FLOATING nllAvg = 0; //Negative log-likelihood averages
  constexpr FLOATING nllWt = 0.1; //Term for weighting the NLL exponential decay

  //Tracker for last-updated term
  vector<int> lastUpdate = vector<int>(nPred);

  for(int test = 0; test < 1; test++){
    BEGIN_TIME();
  uint64_t iterNum = 0; //Iteration counter
  for(int i = 0; i < pred.rows(); i++){
    //Calculate the values needed for the gradient

    BetaVec predSamp = pred.row(i);
    FLOATING exponent = predSamp.dot(guess);
    FLOATING w = 1.0 / (1.0 + exp(-exponent));
    FLOATING y = r(i);
    FLOATING logitDelta = y - m * w;

    //Update tracking negative log likelihood estimate and append to estimate
    FLOATING pointContribNLL = y * log(w) + (m-y) * log(1-w);
    nllAvg = nllAvg * nllWt + (1 - nllAvg) * pointContribNLL;
    objTracker(iterNum) = nllAvg;


    for(BetaVec::InnerIterator it(predSamp); it; ++it){
      int j = it.index();

      // Calculate the L2 penalty term for this element based on last-use time
      FLOATING skip = iterNum - lastUpdate[j];
      FLOATING l2Delta = regCoeff * skip;
      lastUpdate[j] = iterNum;

      // Calculate gradient for this element
      FLOATING elem_gradient = -(l2Delta + logitDelta) * it.value();

      // Update weights for Adagrad
      agWeights(j) += elem_gradient * elem_gradient;
      FLOATING h = invSqrt(agWeights(i) + adagradEpsilon);

      // Scale element
      FLOATING scaleFactor = masterStepSize * h;

      guess(j) -= scaleFactor * elem_gradient;
    }
  }
  END_TIME(t1);
  iterNum++;
  cout << t1 << "  " << t2 << "  " << t3 << endl;
  }

  return objTracker;
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
  DenseVec guess = DenseVec::Constant(predictors.cols(), 0.0);
  DenseVec nll_trac = sgd_iteration(predictors, responses, guess);

  cout << guess << endl;
}
