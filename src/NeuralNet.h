#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include <map>


#include <Eigen/Dense>

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

class NN {
public:
    NN (std::map<std::string, std::string> config, std::string initializer);

    void train();

    void test();

private:
    void rng_initialization(MatrixSingle &matrix, std::string type);

    void shuffle_data(MatrixSingle& tr_i, MatrixSingle& tr_l, MatrixSingle& s_i, MatrixSingle& s_l, std::vector<int>& index) ;

    void forward();
    void backward();
    
    int train_count;
    int test_count;
    int total_count;
    int num_epochs;
    int batch_size;
    int hidden_size;
    int num_classes;
    float eta;
    MatrixSingle w1, w2;
    MatrixSingle tr_i, te_i;
    MatrixInteger tr_l, te_l;
    


};