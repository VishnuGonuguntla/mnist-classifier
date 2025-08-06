#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <chrono>
#include "NeuralNet.h"
#include "helper.h"
#include "DataReader.h"

using namespace std;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::MatrixXd;
using Eigen::all;
// using Eigen::seq;

int source = 60000;
int test = 10000;
int n = 784;
int k = 10;
int q = 1;

int main(int argc, char* argv[]) {
  auto start= chrono::steady_clock::now();
  string config = argv[1];

  Helper init;
  
  map <string,string> parse_data = init.parse(config);
  int num_epochs =  stoi(parse_data["num_epochs"]);
  int batch = stoi(parse_data["batch_size"]);
  int m = stoi(parse_data["hidden_size"]);
  float eta = stoi(parse_data["learning_rate"]);
  string rel_path_train_images = parse_data["rel_path_train_images"];
  string rel_path_train_labels = parse_data["rel_path_train_labels"];
  string rel_path_test_images = parse_data["rel_path_test_images"];
  string rel_path_test_labels = parse_data["rel_path_test_labels"];
  string rel_path_log_file = parse_data["rel_path_log_file"];
  DataReader reader(rel_path_train_images, rel_path_test_images);
  NN network(source, test, n, batch,k, eta);
  MatrixXd tr_i(source,n),te_i(test,n),w1(n+1,m),w2(m+1,k),tr_l(source,k),te_l(source,k);
  MatrixXi true_label(test,q);
  reader.src_data_reading(tr_i, tr_l, rel_path_train_images, rel_path_train_labels,true_label, "train");
  reader.src_data_reading(te_i,te_l, rel_path_test_images, rel_path_test_labels,true_label, "test");
  init.rng_initialization(w1,"he");
  init.rng_initialization(w2,"gloriot");
  network.training(tr_i,tr_l,w1,w2,num_epochs,batch,m,eta);
  network.testing(w1,w2,te_i,te_l,true_label,num_epochs,batch,m,eta);
  auto end=chrono::steady_clock::now();
  cout << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << "\n";
  return 0;
}

