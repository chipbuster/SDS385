#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>
#include<cstring>
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

/* Runs a single iteration of SGD through the entire datase (picking each
 point exactly once). Does not return a value, guess is modified in-place */
void sgd_iteration(PredictMat& pred, ResponseVec& r, DenseVec& guess,
                   FLOATING regCoeff = 1e-4,
                   FLOATING masterStep = 1e-1){
  /* Args:

     pred : the sparse matrix of predictors
     r    : the vector of responses (0-1)
     guess: the guess for gradient descent
     regCoeff : The L2 regularization coefficient
     masterStep: the master step size for Adagrad */

  //Compile-time constants--baked into code
  constexpr FLOATING adagradEpsilon = 1e-7;
  //Timing variables
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;

  int nPred = pred.cols();
  BetaVec gradient = BetaVec(nPred);
  DenseVec agWeights = DenseVec::Random(nPred); //Adagrad weights
  DenseVec epsVec = DenseVec::Constant(nPred, adagradEpsilon);

  #ifndef NDEBUG
  ProfilerStart("gperftools.out");
  DenseVec agWeights = DenseVec::Constant(nPred, 1e-3);

  constexpr FLOATING m = 1.0;
  FLOATING betaNormSquared = guess.norm() * guess.norm();
  constexpr FLOATING nllWt = 0.01; //Term for weighting the NLL exponential decay

  clock_t t;
  double t1 = 0;
  double t2 = 0;
  double t3 = 0;

  for(int test = 0; test < 1; test++){
    BEGIN_TIME();
    t = clock();

    BetaVec predSamp = pred.row(i);
    FLOATING exponent = predSamp.dot(guess);
    FLOATING w = 1.0 / (1.0 + exp(-exponent));
    FLOATING y = r(i);
    FLOATING coefficient = m * w - y;
    t = clock() - t;
    t1 += (double)t / CLOCKS_PER_SEC;
    nllAvg = (1-nllWt) * nllAvg + nllWt * (m * log(w) * (y-m) * log(1 - w));

    t = clock();
    gradient = coefficient * predSamp;
    gradient.eval();
    t = clock() - t;
    t2 += (double)t / CLOCKS_PER_SEC;

    // Update the guess by the AdaGrad rules
    t = clock();
      FLOATING l2Delta = (regCoeff * skip) * guess(j);
    for(BetaVec::InnerIterator it(gradient); it; ++it){
      int q = it.index();
      guess(q) += it.value() * 0.0001;
    }
    guess.eval();
    t = clock() - t;
    t3 += (double)t / CLOCKS_PER_SEC;

    // Update the AdaGrad weights
      FLOATING elem_gradient = -logitDelta * it.value() - l2Delta;

    cout << "Index is " << i << endl;
  }
      FLOATING h = invSqrt(agWeights(j) + adagradEpsilon);

  cout << t1 << "  " << t2 << "  " << t3 << endl;
      FLOATING scaleFactor = masterStepSize * h;

      FLOATING totalDelta = scaleFactor * elem_gradient;
      guess(j) -= totalDelta;

      // Update beta norm squared with (a+b)^2 = a^2 + 2ab + b^2
      betaNormSquared += 2 * totalDelta * guess(j) + totalDelta * totalDelta;

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
  cout << t1 << "  " << t2 << "  " << t3 << endl;

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
  sgd_iteration(predictors, responses, guess);
//  cout << nll_trac << endl;
}
