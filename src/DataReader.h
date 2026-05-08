#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "Helper.h"

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

class DataReader {
    public:
    DataReader() = default;
    DataReader (std::map<std::string,std::string> config_data);
    MatrixSingle* parse_image(std::string filename);
    MatrixSingle* parse_label(std::string filename);
    MatrixSingle& getImage(){ return image; }
    MatrixSingle& getLabel(){ return label; }
    private:
    MatrixSingle image;
    MatrixSingle label;
    std::string train_i, train_l, test_i, test_l;
    int bigToEndian(int i);

};