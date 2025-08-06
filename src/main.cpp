#include <iostream>
#include <map>
#include <chrono>
#include "../include/NeuralNet.h"
#include "../include/helper.h"
#include "../include/DataReader.h"

using namespace std;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::MatrixXd;
using Eigen::all;
using Eigen::seq;

int source = 60000;
int test = 10000;
int n = 784;
int k = 10;
int q = 1;

int main(int argc, char* argv[]) {
  auto start= chrono::steady_clock::now();
  string config = argv[1];

  Helper init;
  
  map<string,string> parse_data = init.parse(config);
  MatrixXd tr_i(source,n),te_i(test,n),w1(n+1,stoi(parse_data["hidden_size"])),w2(stoi(parse_data["hidden_size"])+1,k);
  MatrixXi tr_l(source,k),te_l(source,k);
  DataReader reader(parse_data);
  vector <int> dataParams = reader.deCompress(tr_i,tr_l,"train");
  dataParams = reader.deCompress(te_i,te_l,"test");

  NN network(dataParams[], test, n, parse_data);
  
  // init.rng_initialization(w1,"he");
  // init.rng_initialization(w2,"gloriot");
  // network.training(tr_i,tr_l,w1,w2);
  // network.testing(w1,w2,te_i,te_l,true_label);
  // auto end=chrono::steady_clock::now();
  // cout << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << "\n";
  return 0;
}

