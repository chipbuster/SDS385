#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<iostream>
#include<cstring>
#include<ctime>
#include<cstdint>
#include<limits>

#include "usertypes.hpp"
#include "read_svmlight.hpp"
#include "tinydir.h"

/*##############################################################################
################################################################################
# Define macros and inline functions to help with later computations/profiling #
################################################################################
##############################################################################*/

//Macros for timing things.
static clock_t t;
#define BEGIN_TIME() t = clock()
#define END_TIME(var) t = clock() - t; var += (double)t / CLOCKS_PER_SEC

//Machine epsilons (used for bumping calcuations)
//constexpr FLOATING machEps = std::numeric_limits<FLOATING>::epsilon();

/* Use Martin Ankerl's fastPow:
   http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
   to provide quick powers for Tikhonov regularization */
inline double fastPrecisePow(double a, double b) {
  // calculate approximation with fraction of the exponent
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  // exponentiation by squaring with the exponent's integer part
  // double r = u.d makes everything much slower, not sure why
  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }

  return r * u.d;
}

/* Using Carmack's magical fast inverse square root function. Need to use different
   magic variables to preserve correctness/performance, and use unions to bitcast
   to avoid potentially undefined behavior.
   See https://en.wikipedia.org/wiki/Fast_inverse_square_root for algorithm. */
#ifdef USE_DOUBLES
union cast_double{ uint64_t asLong; double asDouble; };
static inline double invSqrt( const double& x )
{ //Stolen from physicsforums
  cast_double caster;
  caster.asDouble = x;
  double xhalf = ( double )0.5 * caster.asDouble;
  caster.asLong = 0x5fe6ec85e7de30daLL - ( caster.asLong >> 1 );
  double y = caster.asDouble;
  y = y * ( ( double )1.5 - xhalf * y * y );
  y = y * ( ( double )1.5 - xhalf * y * y ); //For better accuracy

  return y;
}
#else
union cast_single{ uint32_t asInt; float asFloat; };
static inline float invSqrt( const float& number )
{ //Stolen from Wikipedia
  cast_single caster;
  constexpr float threehalfs = 1.5F;

  float x2 = number * 0.5F;
  caster.asFloat  = number;
  caster.asInt  = 0x5f3759df - ( caster.asInt >> 1 ); // what the fuck?
  float y  = caster.asFloat;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
#endif

FLOATING calc_all_likelihood(PredictMat& pred, ResponseVec& r, DenseVec& guess){
  /* A function to calculate the entire likelihood function. Probably slow as
     all hell and CERTAINLY not designed to be used in production. */

  DenseVec outp = pred * guess;
  cout << outp.sum() << endl;
  #pragma omp parallel for
  for(int j = 0; j < pred.rows(); j++){
    outp(j) = 1.0 / (1.0 + exp(-outp(j))); // Calculate expit function
    outp(j) = r(j) * log(outp(j)) + (1 - r(j)) * log(1 - outp(j) + 1e-9);
  }

  return -outp.sum();
}

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
     regCoeff : The L2 regularization coefficient (lambda)
     masterStep: the master step size for Adagrad */

  //Compile-time constants--baked into code
  constexpr FLOATING adagradEpsilon = 1e-7;
  constexpr FLOATING m = 1.0;

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

  //Tracker for last-updated term. Used to do lazy updates of the Tikhonov reg.
  vector<int> lastUpdate = vector<int>(nPred);

  for(int iterNum = 0; iterNum < nSamp; iterNum++){
    //Calculate the values needed for the gradient
    BetaVec predSamp = pred.row(iterNum);
    FLOATING exponent = predSamp.dot(guess);
    FLOATING w = 1.0 / (1.0 + exp(-exponent));
    FLOATING y = r(iterNum);
    FLOATING logitDelta = m * w - y;

    //Update tracking negative log likelihood estimate and append to estimate
    // nllAvg = calc_all_likelihood(pred, r, guess); // Uncomment for full negloglik
    nllAvg = nllWt * nllAvg + ((1 - nllWt) * ( y * log(w) + (m - y) * log(1 - w) ));
    objTracker(iterNum) = -nllAvg;

    /* In Eigen 3.2.9, it is very difficult to keep sparse vectors sparse--they
       keep turning into dense vectors, killing performance. To solve this, we
       explicitly use dense vectors, but only iterate over their nonzero components
       using an InnerIterator on the sample point. This is the key to achieving
       speed for this program.
    */

    /* In theory, we need to update every single term of the estimate every
       iteration to account for the L2 regularization term. However, this
       is quite wasteful, since it means having to iterate over 2.4mil elements.
       Instead, we defer updates until the element is accessed again, e.g. if
       the element is accessed on iteration 5 and again on iteration 30, we apply
       25 iterations of regularization all at once on iteration 30. This also helps
       speed things up, and some thought will show that the effect is almost
       identical, aside from the negative log-likelihood calculations.
    */

    for(BetaVec::InnerIterator it(predSamp); it; ++it){
      int j = it.index();

      // Deferred L2 updates, see comment above this for-loop
      FLOATING skip = iterNum - lastUpdate[j];
      FLOATING l2Penalty = regCoeff * skip * guess(j);
      lastUpdate[j] = iterNum;

      // Calculate gradient(j), this element of the gradient
      FLOATING elem_gradient = -logitDelta * it.value() - l2Penalty;

      // Update weights for Adagrad
      agWeights(j) += elem_gradient * elem_gradient;

      // Calculate the scaling factor using fast-inverse-square-root
      FLOATING h = invSqrt(agWeights(j) + adagradEpsilon);

      FLOATING scaleFactor = masterStepSize * h;
      FLOATING totalDelta = scaleFactor * elem_gradient;
      guess(j) += totalDelta; //Update this element

      // Update beta norm squared with (a+b)^2 = a^2 + 2ab + b^2
      betaNormSquared += 2 * totalDelta * guess(j) + totalDelta * totalDelta;
    }

    //    cout << "On IterNum " << iterNum << ", NLLAvg is " << nllAvg << endl;
  }

  // Apply any ridge-regression penalties that we have not yet evaluated
  for(int j = 0; j < nPred; j++){
    FLOATING skip = nSamp - lastUpdate[j];
    FLOATING l2Penalty = regCoeff * skip * guess(j);
    FLOATING h = invSqrt(agWeights(j) + adagradEpsilon);
    FLOATING scaleFactor = masterStepSize * h;
    FLOATING totalDelta = scaleFactor * l2Penalty;
    guess(j) -= totalDelta;
  }

  return objTracker;
}

int main(int argc,char** argv){
  tinydir_dir dir;
  tinydir_file file;
  string dirname = string(argv[1]);
  tinydir_open(&dir, dirname.c_str());

  vector<string> filenames; //A list of files we will (eventually) parse

  if(argc != 2){
    cout << "Usage: " << argv[0] << " <path-to-svmlight-directory>" << endl;
    return -1;
  }

  // Read all the svm files in from the given directory
  // See https://github.com/cxong/tinydir for details

  while (dir.has_next){
    tinydir_readfile(&dir, &file); //Read the current file
    tinydir_next(&dir);          //Get the next file
    if(file.is_dir) continue; //Not interested in directories

    //Does this file name end in '.svm'? If so, add it to our list. If not,
    //we don't care about it and we skip over it.
    size_t len = strlen(file.name);
    if(len < 4                 ||
       file.name[len-1] != 'm' ||
       file.name[len-2] != 'v' ||
       file.name[len-3] != 's' ||
       file.name[len-4] != '.'){
      continue;
    }
    filenames.push_back(dirname + string(file.name));
  }

  tinydir_close(&dir); // Gotta clean up!

  // Parse the files that we read in and generate predictors/responses
  vector<Entry> results = readFileList(filenames);
  std::pair<ResponseVec, PredictMat> out = genPredictors(results);
  cout << "Matrices generated. " << endl;

  ResponseVec responses = out.first;
  PredictMat predictors = out.second;


  t = clock();
  //Create an initial guess and run a single SGD iteration
  DenseVec guess = DenseVec::Constant(predictors.cols(), 0.0);
  DenseVec nll_trac = sgd_iteration(predictors, responses, guess);

  double time_taken = static_cast<double>(clock()  - t) / CLOCKS_PER_SEC;
  cout << "After matrix generation, an SGD pass took " << time_taken << "s" << endl;
  cout << "Writing negative log-likelihood history to 'nll_tracker.csv'" << endl;

  std::ofstream outfile("nll_tracker.csv");

  for (int i = 0; i < nll_trac.size(); i++){
    if(i != 0){ outfile << ",";}
    outfile << nll_trac(i);
  }
  outfile << endl;
}
