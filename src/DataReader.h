#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <Eigen/Dense>

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

class DataReader {
    public:
    DataReader() = default;
    DataReader (std::map<std::string,std::string> config_data);
    std::vector<int> parse_image(std::string filename);
    std::vector<int> parse_label(std::string filename);
    private:
    MatrixSingle image;
    MatrixInteger label;
    std::string train_i, train_l, test_i, test_l;
    int bigToEndian(int i);

};