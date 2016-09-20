#include<fstream>
#include<vector>
#include<sstream>
#include<string>
#include<memory>

using namespace std;

/* A predictor is a fieldnum/val pair which gives the value of a predictor
   in a particular field. In SVMLight this is represented as fieldnum:value */
struct Predictor{
  long fieldnum;
  double value;

  Predictor(long f, double v);
};

typedef vector<Predictor> PredictorList;

/* An entry is a single line of an SVMLight file--it represents a single data
   point. It is an outcome (+1/-1) and a set of predictors */
struct Entry{
  bool outcome;
  PredictorList predictors;

  Entry(bool o, PredictorList p);
};

/* Generate a single Entry from a single line worth of text */
Entry parseSVMLightLine ( char* lineValue);

/* Read a single file and generate a vector of Entries */
vector<Entry> readSVMLightFile(const char* filename );

/* Read a series of files, concatenating them into a giant
   vector at the end of the procedure. This is the main function
   to call in this program. */
vector<Entry> readFileList( int , char** );

std::ostream& operator<<(std::ostream&, Predictor const&);

std::ostream& operator<<(std::ostream&, Entry const&);
