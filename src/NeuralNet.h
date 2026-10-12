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

    void train(const MatrixSingle& image,const  MatrixSingle& label);

    void test(const MatrixSingle& image,const MatrixSingle& label);

private:
    void rng_initialization(MatrixSingle &matrix, std::string type);

    void shuffle_data(Eigen::VectorXi& index) ;
    
    int num_epochs;
    int batch_size;
    int hidden_size;
    long train_count;
    long test_count;
    long num_features;
    long num_classes;
    float eta;
    std::string initializer;
    MatrixSingle w1, w2;

};