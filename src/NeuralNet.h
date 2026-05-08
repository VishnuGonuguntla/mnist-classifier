#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>
#include "Helper.h"

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

using namespace Eigen;

class NN {
public:
    NN (std::map<std::string, std::string> config, std::string initializer);

    void train(MatrixSingle& image, MatrixSingle& label);

    void test(MatrixSingle& image, MatrixSingle& label);

private:
    void rng_initialization(MatrixSingle &matrix, std::string type);

    void shuffle_data(Eigen::VectorXi& index) ;
    
    int num_epochs;
    int batch_size;
    int hidden_size;
    int train_count;
    int test_count;
    int num_features;
    int num_classes;
    float eta;
    std::string initializer;
    MatrixSingle w1, w2;

};