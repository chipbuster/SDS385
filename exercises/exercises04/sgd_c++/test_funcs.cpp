#include "read_svmlight.hpp"

#include<iostream>
#include<cstring>
#include<cassert>

int main(int argc, char** argv){

  if (argc == 1){
    std::cout << "Usage: " << argv[0] << " <svm file names> " << std::endl << std::endl;
    std::cout << "Summarizes the number of entries and sparsity of the input found" << endl
              << "in the specified SVMLight files." << endl;
    return -1;
  }

  char** files = argv + 1;
  vector<Entry> results = readFileList(argc - 1, files);

  std::pair<ResponseVec, PredictMat> out = genPredictors(results);

  ResponseVec vec = out.first;
  PredictMat mat = out.second;

  //Sanity checks
  assert(vec.rows() == mat.rows());

  double matSize = static_cast<double>(mat.rows()) * static_cast<double>(mat.cols());
  cout << "Your data had " << mat.rows() << " entries with " << mat.cols() << " predictors" << endl;
  cout << "The matrix has " << mat.nonZeros() << " nonzero elements for a sparsity of "
       << static_cast<double>(mat.nonZeros()) / matSize << endl;
  cout << "The predictors were " << vec.mean() * 100 << " percent positive (=1)" << endl;

  return 0;
}
