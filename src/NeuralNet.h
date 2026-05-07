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

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

using namespace Eigen;

class NN {
public:
    NN (std::map<std::string, std::string> config, std::string initializer);

    void train(MatrixSingle& tr_i, MatrixInteger& tr_l);

    void test(MatrixSingle& te_i, MatrixInteger& te_l);

private:
    void rng_initialization(MatrixSingle &matrix, std::string type);

    void shuffle_data(Eigen::VectorXi& index) ;

    void forward(MatrixSingle& image, MatrixInteger& label, Eigen::VectorXi& index);
    // void backward();
    
    int train_count;
    int test_count;
    int total_count;
    int num_epochs;
    int batch_size;
    int hidden_size;
    int num_classes;
    float eta;
    std::string initializer;
    MatrixSingle w1, w2;
    MatrixSingle tr_i, te_i;
    MatrixInteger tr_l, te_l;
    
    MatrixSingle s_i, a1, a2, h1, h2;

};