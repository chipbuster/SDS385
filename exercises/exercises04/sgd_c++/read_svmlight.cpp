#include "read_svmlight.hpp"
#include<cstring> //For strtok
#include<cstdlib>
#include<cassert>
#include<iostream> //Mostly for debugging
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

/* See read_svmlight.hpp for general explanations of these functions */

using namespace std;

/* This is mostly support code for our classes*/

Predictor::Predictor(long f, double v){
  this->fieldnum = f;
  this->value = v;
}

Entry::Entry(bool o, PredictorList p){
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

Entry parseSVMLightLine(char* input){
  vector<string> tokens;
  vector<Predictor> preds;
  char* saveptr;

  // Split the string on whitespace and examine each token individually
  char* p = strtok_r(input, " ", &saveptr);
  while(p){
    tokens.push_back(string(p));
    p = strtok_r(NULL, " ", &saveptr);
  }

  auto it = tokens.begin();

  // Examine the first entry of tokens--this should be the response value.
  bool response = (*it == string("1")) ? true : false;
  it++;

  //Loop over remaining entries and process them
  while(it != tokens.end()){
    char* tok = const_cast<char*>(it->c_str());

    // Get first entry in token, it is the fieldnum
    char* fst = strtok_r(tok,":",&saveptr);
    long fieldnum = atoi(fst);

    // Get second entry in token, it is the value of the field
    char* snd = strtok_r(NULL,":",&saveptr);
    double value = atof(snd);

    assert(strtok_r(NULL, ";", &saveptr) == NULL && "STRTOK did not chomp entire entry!");

    preds.push_back(Predictor(fieldnum,value));
    it++;
  }

  Entry retval = Entry(response, preds);
  return retval;
}

vector<Entry> readSVMLightFile(const char* filename){
  string line;
  ifstream infile(filename);
  vector<Entry> entries;

  cout << "Processing " << filename << endl;

  while(getline(infile, line)){

    //Check if the line is commented or blank
    size_t first_nonspace_pos = line.find_first_not_of(" \t\n");
    if(first_nonspace_pos == string::npos){
      cout << "[WARN]: Found a blank line in file " << filename << endl;
      continue; //This is a blank line, apparently...
    }
    if (line[first_nonspace_pos] == '#'){
      cout << "[WARN]: Found a commented line in file  " << filename << endl;
      continue; //This is a commented line
    }

    //Process this line
    entries.push_back( parseSVMLightLine( const_cast<char*>(line.c_str()) ) );
  }

  return entries;
}

vector<Entry> readFileList(int numFiles, char** filenameList){

  vector<vector<Entry> > fileEntries;
  vector<Entry> allEntries;

  // Read the entries from each file and place in a record
  #pragma omp parallel for
  for(int j = 0; j < numFiles; j++){
    vector<Entry> tmp = readSVMLightFile(filenameList[j]);
    #pragma omp critical
    fileEntries.push_back(tmp);
  }

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
