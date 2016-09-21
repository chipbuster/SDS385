#ifndef READ_SVMLIGHT_H
#define READ_SVMLIGHT_H

#include<fstream>
#include<vector>
#include<sstream>
#include<string>
#include<memory>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<utility>
#include<cstdint>

// For PredictMat and Responsevec
#include "usertypes.hpp"

using namespace std;

/* A predictor is a fieldnum/val pair which gives the value of a predictor
   in a particular field. In SVMLight this is represented as fieldnum:value */
struct Predictor{
  uint32_t fieldnum;
  FLOATING value;

  explicit Predictor(uint32_t f, FLOATING v);
  explicit Predictor(const char* s);
};

/* An entry is a single line of an SVMLight file--it represents a single data
   point. It is an outcome (+1/-1) and a set of predictors */
struct Entry{
  FLOATING outcome;
  vector<Predictor> predictors;

  Entry(FLOATING o, vector<Predictor> p);
};

Entry parseSVMLightLine ( char* lineValue);

vector<Entry> readSVMLightFile(const char* filename );

vector<Entry> readFileList( int , char** );

/* Turn a giant vector of Entries into a sparse matrix and a vector */
std::pair<ResponseVec, PredictMat> genPredictors(vector<Entry> allEntries);

std::ostream& operator<<(std::ostream&, Predictor const&);

std::ostream& operator<<(std::ostream&, Entry const&);

#endif
