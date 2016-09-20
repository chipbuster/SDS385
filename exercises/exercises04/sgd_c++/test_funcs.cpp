#include "read_svmlight.hpp"

#include<iostream>
#include<cstring>

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
  readFileList(argc - 1, files);

  return 0;
}
