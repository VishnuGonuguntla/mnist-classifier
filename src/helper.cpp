#include "helper.h"
#include <iostream>
#include <fstream>

using namespace std;

void Helper::rng_initialization(MatrixXd& matrix,string type){ // Initialization: Generates the Initial values for the trainable parameters. Mersenne Twister Engine.
  int rows = matrix.rows()-1;
  random_device rd;
  mt19937 gen( rd() );
  double stddev;
  if (type == "he"){
    stddev = sqrt(2.0/rows);
  }
  else if (type == "gloriot"){
    stddev = sqrt(1.0/rows);
  }
  normal_distribution<double> dist(0, stddev);
  for( int i = 0; i < matrix.rows(); i++ ) {
    for( int j = 0; j < matrix.cols(); j++ ) {
      matrix(i,j) = dist(gen);
    }
  }
}

map<string,string> Helper::parse(string config_file) {
  map<string,string> parse_data;
  ifstream file(config_file);
  string line;
  if (!file.is_open()) {
    cout << "Error opening file " << config_file << endl;
    return parse_data;
  }
  while(getline(file,line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t pos = line.find_first_of(" = ");
    int i = 0;
    if (pos!= string::npos) {
      string key = line.substr(0, pos);
      string value = line.substr(pos+3);
      parse_data[key] = value;
    }
  }
  return parse_data;
}