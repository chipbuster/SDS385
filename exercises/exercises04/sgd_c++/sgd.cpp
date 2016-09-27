#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>
#include<cstring>
#include<ctime>
#include<cstdint>

#include "usertypes.hpp"
#include "read_svmlight.hpp"
#include "tinydir.h"

/*##############################################################################
################################################################################
# Define macros and inline functions to help with later computations/profiling #
################################################################################
##############################################################################*/

#define BEGIN_TIME() t = clock()
#define END_TIME(var) t = clock() - t; var += (double)t / CLOCKS_PER_SEC

//Using Carmack's magical fast inverse square root function
#ifdef USE_DOUBLES
union cast_double{ uint64_t asLong; double asDouble; };
static inline double invSqrt( const double& x )
{ //Stolen from physicsforums
  cast_double caster;
  caster.asDouble = x;
  double xhalf = ( double )0.5 * caster.asDouble;
  caster.asLong = 0x5fe6ec85e7de30daLL - ( caster.asLong >> 1 );//LL suffix for (long long) type for GCC
  double y = caster.asDouble;
  y = y * ( ( double )1.5 - xhalf * y * y );
  y = y * ( ( double )1.5 - xhalf * y * y ); //For better accuracy

  return y;
}
#else
union cast_single{ uint32_t asInt; float asFloat; };
static inline float Q_rsqrt( const float& number )
{ //Stolen from Wikipedia
  cast_single caster;
	constexpr float threehalfs = 1.5F;

	float x2 = number * 0.5F;
	caster.asFloat  = number;
	caster.asInt  = 0x5f3759df - ( caster.asInt >> 1 );               // what the fuck?
	float y  = caster.asFloat;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
#endif

/* #############################################################################
   ###################### BEGIN CODE ###########################################
   ###########################################################################*/

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

  //Timing variables
  clock_t t;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;

  int nPred = pred.cols();
  int nSamp = pred.rows();

  //Adagrad weights--the zeroeth element is for the intercept term, so all
  //accesses for elements need to be offset by one.
  DenseVec agWeights = DenseVec::Constant(nPred, 1e-3);

  //Tracker for the value of the objective function (here, nll_avg)
  DenseVec objTracker = DenseVec::Zero(nPred);
  FLOATING nllAvg = 0; //Negative log-likelihood averages
  FLOATING betaNormSquared = guess.norm() * guess.norm();
  constexpr FLOATING nllWt = 0.01; //Term for weighting the NLL exponential decay

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
    nllAvg = (1-nllWt) * nllAvg + nllWt * (m * log(w) * (y-m) * log(1 - w));
    objTracker(iterNum) = nllAvg;

    for(BetaVec::InnerIterator it(predSamp); it; ++it){
      int j = it.index();

      // Calculate the L2 penalty term for this element based on last-use time
      FLOATING skip = iterNum - lastUpdate[j];
      FLOATING l2Delta = (regCoeff * skip) * guess(j);
      lastUpdate[j] = iterNum;

      // Calculate gradient for this element
      FLOATING elem_gradient = -logitDelta * it.value() - l2Delta;

      // Update weights for Adagrad
      agWeights(j) += elem_gradient * elem_gradient;
      FLOATING h = invSqrt(agWeights(j) + adagradEpsilon);

      // Scale element
      FLOATING scaleFactor = masterStepSize * h;

      FLOATING totalDelta = scaleFactor * elem_gradient;
      guess(j) -= totalDelta;

      // Update beta norm squared with (a+b)^2 = a^2 + 2ab + b^2
      betaNormSquared += 2 * totalDelta * guess(j) + totalDelta * totalDelta;
    }
  }

  // Apply any ridge-regression penalties that we have not yet evaluated
  #ifdef USE_OPENMP
  #pragma omp parallel for
  #endif
  for(int j = 0; j < nPred; j++){
    FLOATING skip = iterNum - lastUpdate[j];
    FLOATING l2Delta = regCoeff * skip * guess(j);
    FLOATING h = invSqrt(agWeights(j) + adagradEpsilon);
    FLOATING scaleFactor = masterStepSize * h;
    FLOATING totalDelta = scaleFactor * l2Delta;
    guess(j) -= totalDelta;
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

  if(argc != 2){
    cout << "Usage: " << argv[0] << " <path-to-svmlight-directory>" << endl;
  }

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

/*
  clock_t test;
double q;
  cout << "Testing fast inverse sqrt vs normal" << endl;
  test = clock();
  long derp = 0;
  for(double z = 0.01; z < 10000; z += 0.0001){
    q = 1.0 / std::sqrt(z);
    derp++;
    if (derp % 100 == 0) cout << q << endl;
  }
  double t1 = (double)(clock() - test) / CLOCKS_PER_SEC;

derp = 0;
  test = clock();
  for(double z = 0.01; z < 10000; z += 0.0001){
    q = invSqrt(z);
    derp++;
    if (derp % 100 == 0) cout << q << endl;
  }
  double t2 = (double)(clock() - test) / CLOCKS_PER_SEC;
  cout << t1 << "  " << t2  << endl;
*/

  // Parse the files
  vector<Entry> results = readFileList(filenames);
  std::pair<ResponseVec, PredictMat> out = genPredictors(results);
  cout << "Matrices generated. " << endl;
  ResponseVec responses = out.first;
  PredictMat predictors = out.second;

  // Do SGD
  DenseVec guess = DenseVec::Constant(predictors.cols(), 0.0);
  DenseVec nll_trac = sgd_iteration(predictors, responses, guess);

//  cout << nll_trac << endl;
}
