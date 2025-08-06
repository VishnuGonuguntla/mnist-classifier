#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>


using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi, Eigen::all, Eigen::seq;

class NN {
    public:
    NN (int source, int test, int n, int b,int k, float eta);


    void training(MatrixXd &tr_i, MatrixXd &tr_l, MatrixXd &w1, MatrixXd &w2, int num_epochs, int batch, int hidden_size, float eta);

    void NN::testing(MatrixXd& w1, MatrixXd& w2, MatrixXd& te_i, MatrixXd& te_l, MatrixXi& true_label,int num_epochs,int batch, int hidden_size,int eta);

    void initialization(MatrixXd& matrix, string type) ;

    void shuffle_data(MatrixXd& tr_i, MatrixXd& tr_l, MatrixXd& s_i, MatrixXd& s_l, vector<int>& index) ;
    private:
    int train_count;
    int test_count;
    int batch_count;
    int total_count;
    int num_classes;
    float learn_rate;
};