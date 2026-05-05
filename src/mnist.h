#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include <Eigen/Dense>

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi


class Mnist {
    std::map<std::string, std::string> configsrc(std::string config_src) {
    void shuffle_data(MatrixXd &image_data, MatrixXd &label_data,
                  std::vector<int> &shuffle_index_data, MatrixXd &shuffled_image,
                  MatrixXd &shuffled_label) {
    void src_data_reading(MatrixXd &image_data, MatrixXd &label_data,
                      std::string train_images, std::string train_labels, std::string type,
                      MatrixXd &test_true);
public:
    void testing(MatrixXd &weighted_matrix_1, MatrixXd &weighted_matrix_2,
             MatrixXd &test_image_data, MatrixXd &test_label_data,
             MatrixXd &test_true_label_data, int batch, std::string output,
             int hidden_size) {

    void training();

};