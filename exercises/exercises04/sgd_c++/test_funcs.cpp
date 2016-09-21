#include "read_svmlight.hpp"

#include<iostream>
#include<cstring>
#include<cassert>

//Use this file by compiling and then running in debugger
//Yes I know it's shitty :(
void test_parseline(string inp){
  parseSVMLightLine(const_cast<char*>(inp.c_str()));
}

int main(int argc, char** argv){

  if (argc == 1){
    std::cout << "Usage: " << argv[0] << " <svm file names> " << std::endl;
  }

  // Testing individual line reader
  //  std::string line;
  //  std::ifstream infile("teststrings.txt");
  //  while(std::getline(infile, line)){
  //   test_parseline(line);
  //  }

  // Test individual file reader
  //auto x = readSVMLightFile("/tmp/url_svmlight/Day1.svm");
  //cout << x.size() << endl;

  // Test directory reader
  char** files = argv + 1;
  auto fl = readFileList(argc - 1, files);

  auto out = genPredictors(fl);

  auto vec = out.first;
  auto mat = out.second;

  //Sanity checks
  assert(vec.rows() == mat.rows());

  cout << "Your data had " << mat.rows() << " entries with " << mat.cols() << " predictors" << endl;
  cout << "The predictors were " << vec.mean() * 100 << " percent positive (=1)" << endl;

  return 0;
}
