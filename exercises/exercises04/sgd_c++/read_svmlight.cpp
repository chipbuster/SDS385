#include "read_svmlight.hpp"
#include<cstring> //For strtok
#include<cstdlib>
#include<cassert>
#include<iostream> //Mostly for debugging
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <cmath>

/* This code is the input-reading code. It ultimately converts a list of files
   into a sparse matrix and dense vector, containing the predictors and
   responses, respectively. To do this, these functions and datatypes are
   implemented:

   ## Datatypes ##
   Predictor: a simple <fieldnum, value> pair. In the SVMLight file, these are
              represented as two numbers separated by a colon, e.g 152:0.23635
              Each sample point consists of a response and multiple predictors.

              Predictor class has a constructor which can (very rapidly)
              initialize its values from a string, e.g. "152:0.23635"

   Entry: an entry is conceptually a single data point, or a single line in an
          SVMLight file. It consists of a response (encoded as 0/1) and a vector
          (or list) of Predictors. An entry will ultimately be split into an
          element of the response vector and the corresponding row of the
          predictor matrix.

   ## Functions ##
   parseSVMLightLine -- Takes in a single line in an SVMLight file and returns an
   Entry. This function does not check if the line is commented or not.

   readSVMLightFile -- Takes in a filename and iterates over all the lines in the
   file, creating an Entry from each by calling parseSVMLightLine, and
   returning a vector of Entries.

   readFileList -- Takes in a list of files and generates a vector of Entries for
   each, then joins them into one giant vector that represents the aggregate of
   all the data points.

   genPredictors -- Takes in a vector of entries representing all the data to be
   analyzed. This vector is then turned into a sparse matrix and a dense vector by
   using Eigen's construction routines.
*/
using namespace std;

/* This is mostly support code for our classes*/

//Construct a predictor from a pair. Very straightforward
Predictor::Predictor(uint32_t f, FLOATING v){
  this->fieldnum = f;
  this->value = v;
}

Predictor::Predictor(const char* s){
  char longfield[16];
  char valfield[32];

  //Search for the colon in the input and record its index such that s[i] == ':'
  size_t i;
  for(i = 0; i < 16; i++){
    if (s[i] == ':'){ break; }
  }
  assert(i <= 11 && "longfield was too long");
  assert(i > 0 && "longfield was not found");

  //Copy the long string and null-terminate it (strncpy) does not do this for us
  strncpy(longfield, s, i);
  longfield[i] = '\0';

  //Copy the float string to valfield array
  strcpy(valfield, (s + i + 1));

  //Read strings into numbers and store
  this->fieldnum = atoi(longfield);
  this->value = atof(valfield);
}

Entry::Entry(FLOATING o, vector<Predictor> p){
  outcome = o;
  predictors = p;
}

std::ostream &operator<<(std::ostream &os, Predictor const &m){
  return os << "(" << m.fieldnum << "," << m.value << ")";
}

std::ostream &operator<<(std::ostream &os, Entry const &m){
  os << m.outcome << ":";
  for (size_t z = 0; z < m.predictors.size(); z++){
    os << " " << m.predictors[z];
  }
  return os;
}


/* #########################################
   ########## Begin actual code ############
   #########################################
*/

/* Takes a single line of an input file and creates an Entry from it */
Entry parseSVMLightLine(char* input){
  vector<string> tokens; // Tokens are individual fields of the file, whitespace-separated
  vector<Predictor> preds;  // One predictor generated per token
  char* saveptr;

  // Split the string on whitespace and store each token to a vector
  char* p = strtok_r(input, " ", &saveptr);
  while(p){
    tokens.push_back(string(p));
    p = strtok_r(NULL, " ", &saveptr);
  }

  auto it = tokens.begin();

  // Examine the first entry of tokens--this should be the response value.
  FLOATING response = (atof(it->c_str()) < 0) ? 0 : 1; //Convert to 0-1
  it++;

  //Loop over remaining entries and turn them into Predictors
  while(it != tokens.end()){
    const char* tok = it->c_str(); //Get the char* from the string for constructor
    preds.push_back(Predictor(tok));
    it++;
  }

  Entry retval = Entry(response, preds);
  return retval;
}

vector<Entry> readSVMLightFile(const char* filename){
  string line;
  ifstream infile(filename);
  vector<Entry> entries;

  // Repeatedly read a line from the file and process it
  while(getline(infile, line)){

    //Check if the line is commented or blank
    size_t first_nonspace_pos = line.find_first_not_of(" \t\n");
    if(first_nonspace_pos == string::npos){
      cerr << "[WARN]: Found a blank line in file " << filename << endl;
      continue; //This is a blank line, we don't need to process it
    }
    if (line[first_nonspace_pos] == '#'){
      cerr << "[WARN]: Found a commented line in file  " << filename << endl;
      continue; //This is a commented line, we shouldn't process it
    }

    //If the line isn't commented or blank, process it!
    entries.push_back( parseSVMLightLine( const_cast<char*>(line.c_str()) ) );
  }

  return entries;
}

vector<Entry> readFileList(int numFiles, char** filenameList){

  vector<vector<Entry> > fileEntries;
  vector<Entry> allEntries;

  // Read the entries from each file and place in a record
#ifdef USE_OPENMP
  #pragma omp parallel for
  for(int j = 0; j < numFiles; j++){
    vector<Entry> tmp = readSVMLightFile(filenameList[j]);
    #pragma omp critical
    fileEntries.push_back(tmp);
  }
#else
  for(int j = 0; j < numFiles; j++){
    vector<Entry> tmp = readSVMLightFile(filenameList[j]);
    fileEntries.push_back(tmp);
  }
#endif


  /* Now we want to flatten fileEntries into one vector, allEntries.
     The best way to do this is to preallocate the memory with reserve()
     and then insert the vectors one at a time with insert() */

  size_t totalsize = 0;
  for(auto& entry : fileEntries){
    totalsize += entry.size();
  }

  allEntries.reserve(totalsize);

  for(auto& entry : fileEntries){
    allEntries.insert(allEntries.end(), entry.begin(), entry.end());
  }

  return allEntries;

}

std::pair<ResponseVec, PredictMat> genPredictors(vector<Entry> allEntries){
  size_t N = allEntries.size();

  //Iterate through the entry list, extracting all the entries
  uint32_t maxFieldnum = 0; //Maximum column index in files, Empirical value: 3231961
  size_t maxPredPerRow = 0; //Maximum number of predictors per row

  // Do a fast pass over the entries to extract statistics (this is quick)
  for(const Entry& ent : allEntries){
    maxPredPerRow = std::max(ent.predictors.size(), maxPredPerRow);

    for(const Predictor& p : ent.predictors){
      maxFieldnum = std::max(maxFieldnum, p.fieldnum);
    }
  }

  //Now we know N, P, and the max# elems per row. Allocate matrices.
  ResponseVec response(N);
  PredictMat preds(N, maxFieldnum);
  preds.reserve(Eigen::VectorXi::Constant(N, maxPredPerRow));

  // Loop over entries and store them into the sparse matrix
  int rowIndex = 0;
  for(const Entry& ent : allEntries){
    const auto& pred = ent.predictors;
    for(const Predictor& p : pred){
      //Insert the entries one-by-one
      preds.insert(rowIndex, p.fieldnum-1) = p.value;
    }
    rowIndex++;
  }

  preds.makeCompressed();

  return std::make_pair(response, preds);
}
